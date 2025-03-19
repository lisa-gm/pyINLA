# Copyright 2024-2025 pyINLA authors. All rights reserved.

import os
from abc import ABC
from pathlib import Path

import numpy as np
from scipy.sparse import spmatrix

from pyinla import ArrayLike, NDArray, sp, xp
from pyinla.configs.likelihood_config import LikelihoodConfig
from pyinla.configs.priorhyperparameters_config import (
    BetaPriorHyperparametersConfig,
    GaussianMVNPriorHyperparametersConfig,
    GaussianPriorHyperparametersConfig,
    PenalizedComplexityPriorHyperparametersConfig,
)
from pyinla.core.likelihood import Likelihood
from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.core.submodel import SubModel
from pyinla.likelihoods import BinomialLikelihood, GaussianLikelihood, PoissonLikelihood
from pyinla.prior_hyperparameters import (
    BetaPriorHyperparameters,
    GaussianMVNPriorHyperparameters,
    GaussianPriorHyperparameters,
    PenalizedComplexityPriorHyperparameters,
)
from pyinla.submodels import (
    BrainiacSubModel,
    RegressionSubModel,
    SpatioTemporalSubModel,
)
from pyinla.utils import scaled_logit


class Model(ABC):
    """Core class for statistical models."""

    def __init__(
        self,
        submodels: list[SubModel],
        likelihood_config: LikelihoodConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""

        # Check the order of the submodels, we want the SpatioTemporalSubModel first
        # as this will decide the sparsity pattern of the precision matrix.
        for i, submodel in enumerate(submodels):
            if isinstance(submodel, SpatioTemporalSubModel):
                submodels.insert(0, submodels.pop(i))

        self.submodels: list[SubModel] = submodels

        # For each submodel...
        theta: ArrayLike = []
        theta_keys: ArrayLike = []
        self.hyperparameters_idx: ArrayLike = [0]
        self.prior_hyperparameters: list[PriorHyperparameters] = []

        for submodel in self.submodels:
            # ...initialize their prior hyperparameters matching their hyperparameters
            if isinstance(submodel, RegressionSubModel):
                ...
            elif isinstance(submodel, SpatioTemporalSubModel):
                # Spatial hyperparameters
                if isinstance(submodel.config.ph_s, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_s,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_s, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_s,
                            hyperparameter_type="r_s",
                        )
                    )

                # Temporal hyperparameters
                if isinstance(submodel.config.ph_t, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_t,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_t, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_t,
                            hyperparameter_type="r_t",
                        )
                    )

                # Sigma spatio-temporal hyperparameters
                if isinstance(
                    submodel.config.ph_st, GaussianPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_st,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_st, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_st,
                            hyperparameter_type="sigma_st",
                        )
                    )
            elif isinstance(submodel, BrainiacSubModel):
                # h2 hyperparameters
                if isinstance(submodel.config.ph_h2, BetaPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        BetaPriorHyperparameters(
                            config=submodel.config.ph_h2,
                        )
                    )

                # alpha hyperparameters
                if isinstance(
                    submodel.config.ph_alpha, GaussianMVNPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        GaussianMVNPriorHyperparameters(
                            config=submodel.config.ph_alpha,
                        )
                    )
            else:
                raise ValueError("Unknown submodel type")

            # ...and read their hyperparameters
            theta_submodel, theta_keys_submodel = submodel.config.read_hyperparameters()

            theta.append(theta_submodel)
            theta_keys += theta_keys_submodel

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_submodel)
            )

        # Add the likelihood hyperparameters
        (
            lh_hyperparameters,
            lh_hyperparameters_keys,
        ) = likelihood_config.read_hyperparameters()

        theta.append(lh_hyperparameters)
        self.theta: NDArray = xp.concatenate(theta)

        theta_keys += lh_hyperparameters_keys
        self.theta_keys: NDArray = theta_keys

        self.n_hyperparameters = self.theta.size

        # --- Initialize the latent parameters and the design matrix
        self.n_latent_parameters: int = 0
        self.latent_parameters_idx: int = [0]

        for submodel in self.submodels:
            self.n_latent_parameters += submodel.n_latent_parameters
            self.latent_parameters_idx.append(self.n_latent_parameters)

        self.x: NDArray = xp.zeros(self.n_latent_parameters)

        data = []
        rows = []
        cols = []
        for i, submodel in enumerate(self.submodels):
            # Convert csc_matrix to coo_matrix to allow slicing
            coo_submodel_a = submodel.a.tocoo()
            data.append(coo_submodel_a.data)
            rows.append(coo_submodel_a.row)
            cols.append(
                coo_submodel_a.col
                + self.latent_parameters_idx[i]
                * xp.ones(coo_submodel_a.col.size, dtype=int)
            )

            self.x[
                self.latent_parameters_idx[i] : self.latent_parameters_idx[i + 1]
            ] = submodel.x_initial

        self.a: spmatrix = sp.sparse.coo_matrix(
            (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
            shape=(submodel.a.shape[0], self.n_latent_parameters),
        )

        # --- Load observation vector
        input_dir = Path(
            kwargs.get("input_dir", os.path.dirname(submodels[0].config.input_dir))
        )

        y: NDArray = np.load(input_dir / "y.npy")
        if xp == np:
            self.y: NDArray = y
        else:
            self.y: NDArray = xp.asarray(y)

        self.y = self.y.flatten()

        self.n_observations: int = self.y.shape[0]

        # --- Initialize likelihood
        # TODO: clean this -> so that for brainiac model we don't add additional hyperperameter
        if likelihood_config.type == "gaussian":
            self.likelihood: Likelihood = GaussianLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )

            if self.submodels[0] == BrainiacSubModel:
                # skip setting prior as it's already set in the submodel
                print(
                    "Brainiac model detected. Skipping setting prior hyperparameters as already set."
                )
            # Instantiate the prior hyperparameters for the likelihood
            elif isinstance(
                likelihood_config.prior_hyperparameters,
                GaussianPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        config=likelihood_config.prior_hyperparameters,
                    )
                )
            elif isinstance(
                likelihood_config.prior_hyperparameters,
                PenalizedComplexityPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparameters(
                        config=likelihood_config.prior_hyperparameters,
                        hyperparameter_type="prec_o",
                    )
                )
        elif likelihood_config.type == "poisson":
            self.likelihood: Likelihood = PoissonLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )
        elif likelihood_config.type == "binomial":
            self.likelihood: Likelihood = BinomialLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )

        self.likelihood_config: LikelihoodConfig = likelihood_config

        # --- Recurrent variables
        self.Q_prior = None
        self.Q_prior_data_mapping = [0]
        self.Q_conditional = None
        self.Q_conditional_data_mapping = [0]

    def construct_Q_prior(self) -> spmatrix:
        kwargs = {}

        if self.Q_prior is None:
            # During the first construction of Q_prior, we allocate the memory for
            # the data and the mapping of each submodel's to the Q prior matrix.
            rows = []
            cols = []
            data = []

            for i, submodel in enumerate(self.submodels):
                if isinstance(submodel, RegressionSubModel):
                    ...
                elif isinstance(submodel, SpatioTemporalSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])
                elif isinstance(submodel, BrainiacSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])

                submodel_Q_prior = submodel.construct_Q_prior(**kwargs)

                rows.append(
                    submodel_Q_prior.row
                    + self.latent_parameters_idx[i] * xp.ones(len(submodel_Q_prior.row))
                )
                cols.append(
                    submodel_Q_prior.col
                    + self.latent_parameters_idx[i] * xp.ones(len(submodel_Q_prior.col))
                )
                data.append(submodel_Q_prior.data)

                self.Q_prior_data_mapping.append(
                    self.Q_prior_data_mapping[i] + len(submodel_Q_prior.data)
                )

            self.Q_prior: sp.sparse.csc_matrix = sp.sparse.csc_matrix(
                (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
                shape=(self.n_latent_parameters, self.n_latent_parameters),
            )

        else:
            for i, submodel in enumerate(self.submodels):
                if isinstance(submodel, RegressionSubModel):
                    ...
                elif isinstance(submodel, SpatioTemporalSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])
                elif isinstance(submodel, BrainiacSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])

                submodel_Q_prior = submodel.construct_Q_prior(**kwargs)

                self.Q_prior.data[
                    self.Q_prior_data_mapping[i] : self.Q_prior_data_mapping[i + 1]
                ] = submodel_Q_prior.data

        return self.Q_prior

    def construct_Q_conditional(
        self,
        eta: NDArray,
    ) -> float:
        """Construct the conditional precision matrix.

        Note
        ----
        Input of the hessian of the likelihood is a diagonal matrix.
        The negative hessian is required, therefore the minus in front.

        """

        # TODO: need to vectorize
        # hessian_likelihood_diag = hessian_diag_finite_difference_5pt(
        #     self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
        # )
        # hessian_likelihood = diags(hessian_likelihood_diag)

        # TODO: needs to be generalized somehow so that we can have multiple likelihoods
        if self.likelihood_config.type == "gaussian":
            kwargs = {
                "eta": eta,
                "theta": float(self.theta[-1]),
            }
        else:
            kwargs = {
                "eta": eta,
            }

        if isinstance(self.submodels[0], BrainiacSubModel):
            # Brainiac specific rule
            kwargs["h2"] = float(self.theta[0])
            d_matrix = self.submodels[0].evaluate_d_matrix(**kwargs)
        else:
            # General rules
            d_matrix = self.likelihood.evaluate_hessian_likelihood(**kwargs)

        self.Q_conditional = self.Q_prior - self.a.T @ d_matrix @ self.a

        return self.Q_conditional

    def construct_information_vector(
        self,
        eta: NDArray,
        x_i: NDArray,
    ) -> NDArray:
        """Construct the information vector."""

        if isinstance(self.submodels[0], BrainiacSubModel):
            kwargs = {"h2": float(self.theta[0])}
            # print("kwargs: ", kwargs)
            gradient_likelihood = self.submodels[0].evaluate_gradient_likelihood(
                eta=eta, y=self.y, **kwargs
            )

        else:
            gradient_likelihood = self.likelihood.evaluate_gradient_likelihood(
                eta=eta,
                y=self.y,
                theta=self.theta[self.hyperparameters_idx[-1] :],
            )

        information_vector: NDArray = (
            -1 * self.Q_prior @ x_i + self.a.T @ gradient_likelihood
        )

        return information_vector

    def evaluate_log_prior_hyperparameters(self) -> float:
        """Evaluate the log prior hyperparameters."""
        log_prior = 0.0

        # if BFGS and model scale differ: rescale -- generalize
        if isinstance(self.submodels[0], BrainiacSubModel):
            #
            theta_interpret = self.theta.copy()
            # print("in evaluate log prior: theta_interpret unscaled: ", theta_interpret)
            theta_interpret[0] = scaled_logit(self.theta[0], direction="backward")
            # print("theta_interpret scaled: ", theta_interpret)
            # TODO: multivariate prior for a ... need to generalize for now:
            log_prior += self.prior_hyperparameters[0].evaluate_log_prior(
                theta_interpret[0]
            )

            log_prior += self.prior_hyperparameters[1].evaluate_log_prior(
                theta_interpret[1:]
            )
        else:
            theta_interpret = self.theta

            for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
                log_prior += prior_hyperparameter.evaluate_log_prior(theta_interpret[i])

        return log_prior

    def get_theta_likelihood(self) -> NDArray:
        """Return the likelihood hyperparameters."""

        if isinstance(self.submodels[0], BrainiacSubModel):
            theta_likelihood = 1 - scaled_logit(self.theta[0], direction="backward")
        else:
            theta_likelihood = self.theta[self.hyperparameters_idx[-1] :]

        return theta_likelihood

    def evaluate_likelihood(self, eta: NDArray, y: NDArray, **kwargs) -> float:
        """Evaluate the likelihood."""

        if isinstance(self.submodels[0], BrainiacSubModel):
            kwargs["h2"] = float(self.theta[0])
            likelihood = self.submodels[0].evaluate_likelihood(eta, y, **kwargs)
        else:
            likelihood = self.likelihood.evaluate_likelihood(eta, y)

        return likelihood

    def __str__(self) -> str:
        """String representation of the model."""
        # Collect general information about the model
        model_info = [
            " --- Model ---",
            f"n_hyperparameters: {self.n_hyperparameters}",
            f"n_latent_parameters: {self.n_latent_parameters}",
            f"n_observations: {self.n_observations}",
            f"likelihood: {self.likelihood_config.type}",
        ]

        # Collect each submodel's information
        submodel_info = [str(submodel) for submodel in self.submodels]

        # Combine model information and submodel information
        return "\n".join(model_info + submodel_info)
