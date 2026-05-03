# Copyright 2024-2026 DALIA authors. All rights reserved.

import re

import numpy as np
from tabulate import tabulate

from dalia import ArrayLike, NDArray, sp, xp
from dalia.configs.models_config import FederatedModelConfig
from dalia.core.model import Model
from dalia.core.prior_hyperparameters import PriorHyperparameters
from dalia.prior_hyperparameters import (
    GaussianMVNPriorHyperparameters,
)
from dalia.submodels import RegressionSubModel
from dalia.submodels.brainiac import BrainiacSubModel
from dalia.utils import (
    add_str_header,
    align_tables_side_by_side,
    bdiag_tiling,
    boxify,
    free_unused_gpu_memory,
)
from dalia.utils.scalar_ndarray import ensure_scalar


class FederatedModel(Model):
    """Federated model class."""

    def __init__(
        self,
        models: list[Model],
        federated_model_config: FederatedModelConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        self.models: list[Model] = models

        # Check the federated type (for now only regression)
        self.federated_type = "regression"
        ## need to check that all submodels have the same number of fixed effects
        ## as they are all estimating the same parameters (for now, later allow custom)

        self.n_models: int = federated_model_config.n_models
        assert self.n_models == len(
            self.models
        ), "Number of models does not match the number of models in the FederatedModelConfig"

        # simply set theta according to first model
        first_model = self.models[0]
        if len(first_model.submodels) != 1:
            raise ValueError("Only one submodel per model is allowed for now.")
        if not isinstance(first_model.submodels[0], RegressionSubModel):
            raise ValueError("For now only regression submodels are allowed.")

        self.n_fixed_effects = first_model.submodels[0].n_fixed_effects
        ref_submodel_type = type(first_model.submodels[0])
        ref_n_latent_parameters = first_model.n_latent_parameters
        ref_n_hyperparameters = first_model.n_hyperparameters

        # Federated config owns the hyperparameter superset.
        theta_federated_config, theta_keys_federated_config = (
            federated_model_config.read_hyperparameters()
        )

        self.prior_hyperparameters: list[PriorHyperparameters] = (
            first_model.prior_hyperparameters
        )
        self.theta_external = theta_federated_config
        self.n_hyperparameters = self.theta_external.size
        self.theta_keys = theta_keys_federated_config
        self.hyperparameters_idx: ArrayLike = first_model.hyperparameters_idx

        self.n_observations: int = 0
        self.n_observations_idx: list[int] = [0]

        ## need to check that all models have the same type of submodel and the same size of submodel
        ## needs to be regression for now
        for i, model in enumerate(self.models):
            if len(model.submodels) != 1:
                raise ValueError(
                    f"Model {model} has more than one submodel. Only one submodel per model is allowed for now."
                )
            if type(model.submodels[0]) is not ref_submodel_type:
                raise ValueError(
                    f"Model {model} has a different submodel type. Expected {ref_submodel_type.__name__}, got {type(model.submodels[0]).__name__}."
                )
            if self.n_fixed_effects != model.submodels[0].n_fixed_effects:
                raise ValueError(
                    f"Model {model} has a different number of fixed effects than allowed. Expected {self.n_fixed_effects}, got {model.submodels[0].n_fixed_effects}."
                )

            if model.n_latent_parameters != ref_n_latent_parameters:
                raise ValueError(
                    f"Model {i} has a different number of latent parameters. "
                    f"Expected {ref_n_latent_parameters}, got {model.n_latent_parameters}."
                )

            if model.n_hyperparameters != ref_n_hyperparameters:
                raise ValueError(
                    f"Model {i} has a different number of hyperparameters. "
                    f"Expected {ref_n_hyperparameters}, got {model.n_hyperparameters}."
                )

            # would be best if number of observations could be kept private to each model?
            # but I also need it to weight the contributions
            self.n_observations += model.n_observations
            self.n_observations_idx.append(self.n_observations)
            print(
                f"Model {i} has {model.n_observations} observations. Total so far: {self.n_observations}"
            )

        self.n_latent_parameters = self.n_fixed_effects

        # private to each model: self.model.y, self.model.a
        # self.model.x shared across all
        self.x: NDArray = xp.zeros(self.n_latent_parameters)
        self.y: NDArray = xp.zeros(self.n_observations)

        ### for compatibility initialize dummy a
        self.a = sp.sparse.csc_matrix((self.n_observations, self.n_fixed_effects))

        # set them to none to make sure they are not used
        for model in self.models:
            model.x = None
            # model.theta_external = None
            # model.theta_internal = None

        self.Q_conditional = None
        self.Q_prior: sp.sparse.spmatrix = (
            None  # need this otherwise the construct will fail
        )

        self.construct_Q_prior()

    # ########################################################################
    # @property
    # def theta_external(self):
    #     """External/user/interpretable scale theta."""
    #     # the copy is important to make sure that in place operations still trigger updating
    #     return self._theta_external.copy()

    # @theta_external.setter
    # def theta_external(self, value):
    #     """Set external theta and automatically update internal.

    #     Notes
    #     -----
    #     The re-scaling is implemented for all prios but PenalizedComplexity (identity but already in the correct "log" scale).
    #     """
    #     self._theta_external = xp.array(value)
    #     self._theta_internal = self.rescale_hyperparameters_to_internal(
    #         self._theta_external, direction="forward"
    #     )

    # @property
    # def theta_internal(self):
    #     """Internal/BFGS scale theta."""
    #     return self._theta_internal.copy()

    # @theta_internal.setter
    # def theta_internal(self, value):
    #     """Set internal theta and automatically update external."""
    #     self._theta_internal = xp.array(value)
    #     self._theta_external = self.rescale_hyperparameters_to_internal(
    #         self._theta_internal, direction="backward"
    #     )

    def construct_Q_prior(self) -> sp.sparse.spmatrix:
        """Construct the prior precision matrix.

        Note
        ----
        The prior is the same for all models, therefore we can simply use the first model to construct it.
        """

        # Qprior is the same for all models
        # simply use the first model to construct it
        self.models[0].theta_external = self.theta_external
        self.Q_prior = self.models[0].construct_Q_prior()

        return self.Q_prior

    def construct_Q_conditional(
        self,
        eta: NDArray,
        x: NDArray = None,
    ) -> float:
        """Construct the conditional precision matrix.

        Note
        ----
        Input of the hessian of the likelihood is a diagonal matrix.
        The negative hessian is required, therefore the minus in front.

        Iteratively add the contributions of each model. Something along the lines of
        Q_conditional = Q_prior + A_1^T D_1 A_1 + A_2^T D_2 A_2 + ... + A_n^T D_n A_n

        """

        self.Q_conditional = self.Q_prior.copy()

        for i, model in enumerate(self.models):
            model.theta_external = self.theta_external

            if x is not None:
                eta_i = model.a @ x
            else:
                eta_i = eta[
                    self.n_observations_idx[i] : self.n_observations_idx[i + 1]
                ]

            # negative hessian, therefore minus in front
            self.Q_conditional -= model.construct_ATDA(eta=eta_i)

        return self.Q_conditional

    def construct_information_vector(
        self,
        eta: NDArray,
        x_i: NDArray,
    ) -> NDArray:
        """Construct the information vector.

        Note
        ----
        Compute information vector for each model and then sum them.
        """

        information_vector = -1 * self.Q_prior @ x_i

        for i, model in enumerate(self.models):
            eta_i = eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]]
            information_vector += (
                model.a.T
                @ model.likelihood.evaluate_gradient_likelihood(
                    eta=eta_i,
                    y=model.y,
                    theta=self.theta_external[self.hyperparameters_idx[-1] :],
                )
            )

        return information_vector

    def is_likelihood_gaussian(self) -> bool:
        """Check if the likelihood is Gaussian.

        Returns
        -------
        is_gaussian : bool
            True if the likelihood is Gaussian, False otherwise.
        """
        for model in self.models:
            if not model.is_likelihood_gaussian():
                return False
        return True

    def evaluate_likelihood(
        self,
        eta: NDArray,
    ) -> float:
        """Evaluate the likelihood.

        Parameters
        ----------
        eta : NDArray
            Linear predictor.
        kwargs : dict
            Additional arguments for the likelihood evaluation. These parameters are model dependent.

        Returns
        -------
        likelihood : float
            The evaluated likelihood.

        Implementation Notes:
        ---------------------
        - The likelihood is evaluated for each model and then summed up to get the total likelihood of the CoregionalModel.
        - Returned as a scalar for consistency, even if the likelihood is computed as a sum of multiple likelihoods from different models.
        """
        likelihood: float = 0.0
        for i, model in enumerate(self.models):
            eta_i = eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]]
            likelihood += model.likelihood.evaluate_likelihood(
                eta=eta_i,
                y=model.y,
                theta=self.theta_external[self.hyperparameters_idx[-1] :],
            )

        return ensure_scalar(likelihood)

    def rescale_hyperparameters_to_internal(self, theta, direction):

        # need to iterate over theta and its prior hyperparameters
        theta_internal = xp.copy(theta)

        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            ## how to handle priors that have multiple hyperparameters?
            if isinstance(prior_hyperparameter, GaussianMVNPriorHyperparameters):
                pass  # no rescaling implemented
            else:
                theta_internal[i] = (
                    prior_hyperparameter.rescale_hyperparameters_to_internal(
                        theta[i], direction=direction
                    )
                )

        return theta_internal

    def evaluate_log_prior_hyperparameters(self) -> float:
        """Evaluate the log prior hyperparameters."""
        log_prior = 0.0

        theta_interpret = self.theta_external

        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            log_prior += prior_hyperparameter.evaluate_log_prior(theta_interpret[i])

        return log_prior

    def __str__(self) -> str:
        """String representation of the model."""
        str_representation = ""

        # --- Make the Coregional Model() table ---
        headers = [
            "Number of Hyperparameters",
            "Number of Latent Parameters",
            "Number of Observations",
        ]
        values = [self.n_hyperparameters, self.n_latent_parameters, self.n_observations]

        model_table = tabulate(
            [headers, values],
            tablefmt="fancy_grid",
            colalign=("center", "center", "center"),
        )

        # Add the header title
        model_table = add_str_header(
            f"Coregional Model ({self.n_models} variates)", model_table
        )

        # --- Add the model information ---
        # Create headers and values for the model table
        models_str_representation = []
        for model in self.models:
            models_str_representation.append(str(model))

        # Create the model table
        model_jointed_representation = align_tables_side_by_side(
            models_str_representation
        )

        # Add the model header title
        model_jointed_representation = add_str_header(
            "Models", model_jointed_representation
        )

        # Combine the model and model tables
        str_representation = model_table + "\n" + boxify(model_jointed_representation)

        return str_representation
