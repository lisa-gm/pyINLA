# Copyright 2024-2026 DALIA authors. All rights reserved.

from tabulate import tabulate

from dalia import ArrayLike, NDArray, sp, xp
from dalia.configs.models_config import ReplicateModelConfig
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


class ReplicateModel(Model):
    """Replicate model class.

    Assumes that we have multiple instances of the same model, each with its own data and covariates, but sharing the same hyperparameters.
    """

    def __init__(
        self,
        models: list[Model],
        replicate_model_config: ReplicateModelConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        self.models: list[Model] = models

        self.n_models: int = replicate_model_config.n_models
        assert self.n_models == len(
            self.models
        ), "Number of models does not match the number of models in the ReplicateModelConfig"

        # simply set theta according to first model
        first_model = self.models[0]

        self.n_latent_effects = self.n_models * first_model.n_latent_effects
        ref_n_submodels = len(first_model.submodels)
        ref_submodel_types = [type(submodel) for submodel in first_model.submodels]

        ref_latent_parameters_idx = list(first_model.latent_parameters_idx)
        ref_hyperparameters_idx = list(first_model.hyperparameters_idx)
        ref_theta_keys = list(first_model.theta_keys)
        ref_likelihood_type = type(first_model.likelihood)
        ref_n_latent_parameters = first_model.n_latent_parameters
        ref_n_hyperparameters = first_model.n_hyperparameters

        # Replicate config owns the hyperparameter superset.
        theta_replicate_config, theta_keys_replicate_config = (
            replicate_model_config.read_hyperparameters()
        )

        self.prior_hyperparameters: list[PriorHyperparameters] = (
            first_model.prior_hyperparameters
        )
        self.theta_external = theta_replicate_config
        self.n_hyperparameters = self.theta_external.size
        self.theta_keys = theta_keys_replicate_config
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

            if model.n_fixed_effects != first_model.n_fixed_effects:
                raise ValueError(
                    f"Model {i} has a different number of fixed effects. "
                    f"Expected {first_model.n_fixed_effects}, got {model.n_fixed_effects}."
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

            # each instance can have a different number of observations
            self.n_observations += model.n_observations
            self.n_observations_idx.append(self.n_observations)

        # this is now the total number of latent parameters (i.e. contains multiple instances of the same latent parameter)
        self.x: NDArray = xp.zeros(self.n_latent_parameters)
        # total number of observations across all replicates
        self.y: NDArray = xp.zeros(self.n_observations)

        ### construct the observation matrix A as a block diagonal matrix of all local models
        self.a = sp.sparse.block_diag([model.a for model in self.models], format="csc")

        # now check that col(a) == n_latent_parameters, row(a) == n_observations
        if self.a.shape[0] != self.n_observations:
            raise ValueError(
                f"Observation matrix A has {self.a.shape[0]} rows, but expected {self.n_observations} (total number of observations across all models)."
            )
        if self.a.shape[1] != self.n_latent_parameters:
            raise ValueError(
                f"Observation matrix A has {self.a.shape[1]} columns, but expected {self.n_latent_parameters} (total number of latent parameters across all models)."
            )

        self.Q_conditional = None
        self.Q_prior: sp.sparse.spmatrix = None

        self.construct_Q_prior()

    def construct_Q_prior(self) -> sp.sparse.spmatrix:
        """Construct the prior precision matrix.

        Note
        ----
        The prior is the same for all models, therefore we can simply use the first model to construct it.
        """

        # create block diagonal matrix of Q_prior from each model
        # can use the first model to construct them all
        self.models[0].theta_external = self.theta_external
        self.Q_prior = sp.sparse.block_diag(
            [self.models[0].construct_Q_prior() for _ in self.models], format="csc"
        )

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
        self.Q_conditional -= self.construct_ATDA(eta=eta)

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
        print(f"n_observations_idx: {self.n_observations_idx}")
        exit()

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
        - The likelihood is evaluated for each model and summed over all models.
        """

        likelihood: float = 0.0
        for _, model in enumerate(self.models):
            # local eta
            eta = model.a @ model.x
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
            "Number of Replicates",
            "Number of Hyperparameters",
            "Total number of Latent Parameters",
            "Total numberof Fixed Effects",
            "Total number of Observations",
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
