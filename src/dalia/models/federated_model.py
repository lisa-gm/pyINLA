# Copyright 2024-2026 DALIA authors. All rights reserved.

from tabulate import tabulate

from dalia import ArrayLike, NDArray, sp, xp
from dalia.configs.models_config import FederatedModelConfig
from dalia.core.model import Model
from dalia.core.prior_hyperparameters import PriorHyperparameters
from dalia.prior_hyperparameters import (
    GaussianMVNPriorHyperparameters,
)
from dalia.utils import (
    add_str_header,
    align_tables_side_by_side,
    boxify,
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

        # Federated model over structurally identical local models
        self.federated_type = "regression"

        self.n_models: int = federated_model_config.n_models
        assert self.n_models == len(
            self.models
        ), "Number of models does not match the number of models in the FederatedModelConfig"

        # simply set theta according to first model
        first_model = self.models[0]

        self.n_fixed_effects = first_model.n_fixed_effects
        ref_n_submodels = len(first_model.submodels)
        ref_submodel_types = [type(submodel) for submodel in first_model.submodels]
        ref_latent_parameters_idx = list(first_model.latent_parameters_idx)
        ref_hyperparameters_idx = list(first_model.hyperparameters_idx)
        ref_theta_keys = list(first_model.theta_keys)
        ref_likelihood_type = type(first_model.likelihood)
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

        ## ensure all local models have the same structure
        for i, model in enumerate(self.models):
            if len(model.submodels) != ref_n_submodels:
                raise ValueError(
                    f"Model {i} has a different number of submodels. "
                    f"Expected {ref_n_submodels}, got {len(model.submodels)}."
                )

            model_submodel_types = [type(submodel) for submodel in model.submodels]
            if model_submodel_types != ref_submodel_types:
                raise ValueError(
                    f"Model {i} has different submodel types/order than the reference model. "
                    f"Expected {[t.__name__ for t in ref_submodel_types]}, got {[t.__name__ for t in model_submodel_types]}."
                )

            if self.n_fixed_effects != model.n_fixed_effects:
                raise ValueError(
                    f"Model {i} has a different number of fixed effects. "
                    f"Expected {self.n_fixed_effects}, got {model.n_fixed_effects}."
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

            if list(model.latent_parameters_idx) != ref_latent_parameters_idx:
                raise ValueError(
                    f"Model {i} has different latent parameter block indices. "
                    f"Expected {ref_latent_parameters_idx}, got {list(model.latent_parameters_idx)}."
                )

            if list(model.hyperparameters_idx) != ref_hyperparameters_idx:
                raise ValueError(
                    f"Model {i} has different hyperparameter block indices. "
                    f"Expected {ref_hyperparameters_idx}, got {list(model.hyperparameters_idx)}."
                )

            if list(model.theta_keys) != ref_theta_keys:
                raise ValueError(
                    f"Model {i} has different theta_keys/order. "
                    f"Expected {ref_theta_keys}, got {list(model.theta_keys)}."
                )

            if type(model.likelihood) is not ref_likelihood_type:
                raise ValueError(
                    f"Model {i} has a different likelihood type. "
                    f"Expected {ref_likelihood_type.__name__}, got {type(model.likelihood).__name__}."
                )

            # would be best if number of observations could be kept private to each model?
            # but I also need it to weight the contributions
            self.n_observations += model.n_observations
            self.n_observations_idx.append(self.n_observations)

        self.n_latent_parameters = ref_n_latent_parameters

        # private to each model: self.model.y, self.model.a
        # self.model.x shared across all
        self.x: NDArray = xp.zeros(self.n_latent_parameters)
        self.y: NDArray = xp.zeros(self.n_observations)

        ### for compatibility initialize dummy a
        self.a = sp.sparse.csc_matrix((self.n_observations, self.n_latent_parameters))

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
            # TODO: move this inside the model and pass only x
            # overwrite eta if x is provided as eta is private in federated setting
            if x is not None:
                eta = model.a @ x
                # print("norm(model.a) in loop: ", xp.linalg.norm(model.a))
                # print("norm(eta) in loop: ", xp.linalg.norm(eta))
            # negative hessian, therefore minus in front
            self.Q_conditional -= model.construct_ATDA(eta=eta)

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
            # TODO:move this inside the model and pass only x
            eta = model.a @ x_i
            information_vector += (
                model.a.T
                @ model.likelihood.evaluate_gradient_likelihood(
                    eta=eta,
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
        x: NDArray = None,
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
            # TODO: move this inside the model and pass only x, also y needs to stay local
            eta = model.a @ x
            likelihood += model.likelihood.evaluate_likelihood(
                eta=eta,
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

        # TODO: do I need this local re-assignment?
        theta_internal = self.theta_internal

        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            log_prior += prior_hyperparameter.evaluate_internal_log_prior(
                theta_internal[i]
            )

        return log_prior

    def __str__(self) -> str:
        """String representation of the model."""
        headers = [
            "Federated Type",
            "Number of Models",
            "Number of Hyperparameters",
            "Number of Latent Parameters",
            "Number of Fixed Effects",
            "Number of Observations",
        ]
        values = [
            self.federated_type,
            self.n_models,
            self.n_hyperparameters,
            self.n_latent_parameters,
            self.n_fixed_effects,
            self.n_observations,
        ]

        federated_table = tabulate(
            [headers, values],
            tablefmt="fancy_grid",
            colalign=("center", "center", "center", "center", "center", "center"),
        )
        federated_table = add_str_header("Federated Model", federated_table)

        models_str_representation = []
        for model in self.models:
            models_str_representation.append(str(model))

        model_jointed_representation = align_tables_side_by_side(
            models_str_representation
        )
        model_jointed_representation = add_str_header(
            "Local Models", model_jointed_representation
        )

        return federated_table + "\n" + boxify(model_jointed_representation)
