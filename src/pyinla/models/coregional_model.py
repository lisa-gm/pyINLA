# Copyright 2024-2025 pyINLA authors. All rights reserved.

import os
import re
from abc import ABC
from pathlib import Path

import numpy as np
from scipy.sparse import spmatrix

from pyinla import ArrayLike, NDArray, sp, xp
from pyinla.core.model import Model


from pyinla.configs.models_config import CoregionalModelConfig
from pyinla.configs.priorhyperparameters_config import (
    GaussianPriorHyperparametersConfig,
    PenalizedComplexityPriorHyperparametersConfig,
)
from pyinla.core.likelihood import Likelihood
from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.core.submodel import SubModel
from pyinla.likelihoods import BinomialLikelihood, GaussianLikelihood, PoissonLikelihood
from pyinla.prior_hyperparameters import (
    GaussianPriorHyperparameters,
    PenalizedComplexityPriorHyperparameters,
)
from pyinla.submodels import (
    RegressionSubModel,
    SpatioTemporalSubModel,
    SpatialSubModel,
)

class CoregionalModel(Model):
    """Core class for statistical models."""

    def __init__(
        self,
        models: list[Model],
        coregional_model_config: CoregionalModelConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""

        self.models: list[Model] = models

        # Get Models() hyperparameters
        theta: ArrayLike = []
        theta_keys: ArrayLike = []
        self.hyperparameters_idx: ArrayLike = [0]

        for model in self.models:
            theta_model = model.theta
            theta_keys_model = model.theta_keys

            # remove the theta that correspond to the "sigma_xx" where x can be whatever
            sigma_indices = [i for i, key in enumerate(theta_keys_model) if re.match(r"sigma_\w+", key)]
            theta_model = [theta for i, theta in enumerate(theta_model) if i not in sigma_indices]
            theta_keys_model = [key for i, key in enumerate(theta_keys_model) if i not in sigma_indices]

            theta.append(theta_model)
            theta_keys.append(theta_keys_model)

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_model)
            )

        # Initialize the Coregional Hyperparameters:
        theta_coregional_model, theta_keys_coregional_model = coregional_model_config.read_hyperparameters()
        theta.append(theta_coregional_model)
        theta_keys.append(theta_keys_coregional_model)

        # Finalize the hyperparameters
        self.theta: NDArray = xp.concatenate(theta)
        self.n_hyperparameters = self.theta.size
        self.theta_keys: NDArray = theta_keys

        # Initialize the Coregional Prior Hyperparameters
        self.prior_hyperparameters: list[PriorHyperparameters] = []
        for ph in coregional_model_config.ph_sigmas:
            if ph.type == "gaussian":
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        config=ph,
                    )
                )
            elif ph.type == "penalized_complexity":
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparameters(
                        config=ph,
                    )
                )
            else:
                raise ValueError(f"Invalid prior hyperparameters type: {ph.type}")
        
        for ph in coregional_model_config.ph_lambdas:
            if ph.type == "gaussian":
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        config=ph,
                    )
                )
            elif ph.type == "penalized_complexity":
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparameters(
                        config=ph,
                    )
                )
            else:
                raise ValueError(f"Invalid prior hyperparameters type: {ph.type}")

        # Construct Coregional Model data from its Models()
        self.n_latent_parameters: int = 0
        self.latent_parameters_idx: list[int] = [0]
        self.n_observations: int = 0
        self.n_observations_idx: list[int] = [0]

        for model in self.models:
            self.n_latent_parameters += model.n_latent_parameters
            self.latent_parameters_idx.append(self.n_latent_parameters)

            self.n_observations += model.n_observations
            self.n_observations_idx.append(self.n_observations)

        self.x: NDArray = xp.zeros(self.n_latent_parameters)
        self.y: NDArray = xp.zeros(self.n_observations)

        a_data = []
        a_rows = []
        a_cols = []

        for i, model in enumerate(self.models):
            coo_model_a = model.a.tocoo()
            a_data.append(coo_model_a.data)
            # a_rows.append(coo_model_a.row)
            a_rows.append(
                coo_model_a.row
                + self.n_observations_idx[i]
                * xp.ones(coo_model_a.row.size, dtype=int)
            )
            a_cols.append(
                coo_model_a.col
                + self.latent_parameters_idx[i]
                * xp.ones(coo_model_a.col.size, dtype=int)
            )

            self.x[
                self.latent_parameters_idx[i] : self.latent_parameters_idx[i + 1]
            ] = model.x

            self.y[
                self.n_observations_idx[i] : self.n_observations_idx[i + 1]
            ] = model.y

        self.a: spmatrix = sp.sparse.coo_matrix(
            (xp.concatenate(a_data), (xp.concatenate(a_rows), xp.concatenate(a_cols))),
            shape=(self.n_observations, self.n_latent_parameters),
        )

        # --- Recurrent variables
        self.Q_prior = None
        self.Q_prior_data_mapping = [0]
        self.Q_conditional = None
        self.Q_conditional_data_mapping = [0]

    def construct_Q_prior(self) -> spmatrix:
        
        # Construct Qprior from models

        # Apply the permutation

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

        # Call the models.construct_Q_conditiona()

        return self.Q_conditional

    def construct_information_vector(
        self,
        eta: NDArray,
        x_i: NDArray,
    ) -> NDArray:
        """Construct the information vector."""

        # TODO: need to vectorize !!
        # gradient_likelihood = gradient_finite_difference_5pt(
        #     self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
        # )
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

        theta_interpret = self.theta

        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            tmp = prior_hyperparameter.evaluate_log_prior(theta_interpret[i])
            print("tmp: ", tmp)
            log_prior += tmp

        return log_prior

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
