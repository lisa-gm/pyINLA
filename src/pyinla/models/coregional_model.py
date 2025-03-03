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

        # Check the coregionalization type (Spacial or SpatioTemporal)
        self.coregionalization_type: str
        self.n_models: int = coregional_model_config.n_models
        self.n_spatial_nodes: int = None
        self.n_temporal_nodes: int = 1
        for i, model in enumerate(self.models):
            if i == 0:
                if isinstance(model.submodels[0], SpatioTemporalSubModel):
                    self.coregionalization_type = "spatio_temporal"
                    self.n_spatial_nodes = model.submodels[0].ns
                    self.n_temporal_nodes = model.submodels[0].nt
                elif isinstance(model.submodels[0], SpatialSubModel):
                    self.coregionalization_type = "spatial"
                    self.n_spatial_nodes = model.submodels[0].ns
                else:
                    raise ValueError(
                        "Invalid model type. Must be 'spatial' or 'spatio-temporal'."
                    )
            else:
                # Check that all models are the same
                if isinstance(model.submodels[0], SpatioTemporalSubModel):
                    if self.coregionalization_type != "spatio_temporal":
                        raise ValueError(
                            f"Model {model} is not of the same type as the first model (SpatioTemporalModel)"
                        )
                    # Check that the size of the SpatioTemporal fields are the same
                    if model.submodels[0].ns != self.n_spatial_nodes or model.submodels[0].nt != self.n_temporal_nodes:
                        raise ValueError(
                            f"Model {model} is not of the same size as the first model (SpatioTemporalModel)"
                        )
                elif isinstance(model.submodels[0], SpatialSubModel):
                    if self.coregionalization_type != "spatial":
                        raise ValueError(
                            f"Model {model} is not of the same type as the first model (SpatialModel)"
                        )
                    # Check that the size of the Spatial fields are the same
                    if model.submodels[0].ns != self.n_spatial_nodes:
                        raise ValueError(
                            f"Model {model} is not of the same size as the first model (SpatialModel)"
                        )

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
            theta_keys += theta_keys_model

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_model)
            )

        # Initialize the Coregional Hyperparameters:
        theta_coregional_model, theta_keys_coregional_model = coregional_model_config.read_hyperparameters()
        theta.append(theta_coregional_model)
        theta_keys += theta_keys_coregional_model

        self.hyperparameters_idx.append(
            self.hyperparameters_idx[-1] + len(theta_coregional_model)
        )

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
        Q_list: list = []

        for i, model in enumerate(self.models):

            Q_list.append(model.construct_Q_prior())

        # coregional_kwargs = {}
        # for hp_idx in range(
        #     self.hyperparameters_idx[-2], self.hyperparameters_idx[-1]
        # ):
        #     coregional_kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])

        # print(f"coregional_kwargs: {coregional_kwargs}")

        # print theta at the index of theta_keys["sigma_0"]
        print(self.theta_keys.index("sigma_0"))
        print(self.theta[self.theta_keys.index("sigma_0")])

        sigma_0 = xp.exp(self.theta[self.theta_keys.index("sigma_0")])
        sigma_1 = xp.exp(self.theta[self.theta_keys.index("sigma_1")])

        lambda_0_1 = self.theta[self.theta_keys.index("lambda_0_1")]

        if self.n_models == 2:
            Q = sp.sparse.vstack(
                [
                    sp.sparse.hstack(
                        [
                            (1 / sigma_0**2) * Q_list[0]
                            + (lambda_0_1**2 / sigma_1**2) * Q_list[1],
                            (-lambda_0_1 / sigma_1**2) * Q_list[1],
                        ]
                    ),
                    sp.sparse.hstack(
                        [
                            (-lambda_0_1 / sigma_1**2) * Q_list[1],
                            (1 / sigma_1**2) * Q_list[1],
                        ]
                    ),
                ]
            )
        elif self.n_models == 3:
            sigma_2 = xp.exp(self.theta[self.theta_keys.index("sigma_2")])

            lambda_0_2 = self.theta[self.theta_keys.index("lambda_0_2")]
            lambda_1_2 = self.theta[self.theta_keys.index("lambda_1_2")]

            Q = sp.sparse.vstack(
                [
                    sp.sparse.hstack(
                        [
                            (1 / sigma_0**2) * Q_list[0]
                            + (lambda_0_1**2 / sigma_1**2) * Q_list[1]
                            + (lambda_1_2**2 / sigma_2**2) * Q_list[2],
                            (-lambda_0_1 / sigma_1**2) * Q_list[1]
                            + (lambda_0_2 * lambda_1_2 / sigma_2**2) * Q_list[2],
                            -lambda_1_2 / sigma_2**2 * Q_list[2],
                        ]
                    ),
                    sp.sparse.hstack(
                        [
                            (-lambda_0_1 / sigma_1**2) * Q_list[1]
                            + (lambda_0_2 * lambda_1_2 / sigma_2**2) * Q_list[2],
                            (1 / sigma_1**2) * Q_list[1]
                            + (lambda_0_2**2 / sigma_2**2) * Q_list[2],
                            -lambda_0_2 / sigma_2**2 * Q_list[2],
                        ]
                    ),
                    sp.sparse.hstack(
                        [
                            -lambda_1_2 / sigma_2**2 * Q_list[2],
                            -lambda_0_2 / sigma_2**2 * Q_list[2],
                            (1 / sigma_2**2) * Q_list[2],
                        ]
                    ),
                ]
            )


        # Apply the permutation
        if self.coregionalization_type == "spatio_temporal":
            # Permute matrix
            p_vec = self._generate_permutation_indices(self.n_temporal_nodes, self.n_spatial_nodes, self.n_models)

            # permute Q
            Qprior_perm1 = Q[p_vec, :][:, p_vec]

        return Qprior_perm1

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

    def _generate_permutation_indices(self, n_temporal_nodes: int, n_spatial_nodes: int, n_models: int):
        """
        Generate a permutation vector containing indices in the pattern:
        [0:block_size, n*block_size:(n+1)*block_size, 1*block_size:(1+1)*block_size, (n+1)*block_size:(n+1+1)*block_size, ...]

        Parameters
        ----------
        n_temporal_nodes : int
            Number of blocks.
        n_spatial_nodes : int
            Size of each block.
        n_models : int
            Number of models.
        
        Returns
        -------
        np.ndarray
            The generated permutation vector.
        """
        indices = np.arange(n_temporal_nodes * n_spatial_nodes)

        first_idx = indices.reshape(n_temporal_nodes, n_spatial_nodes)
        second_idx = first_idx + n_temporal_nodes * n_spatial_nodes

        if n_models == 2:
            perm_vectorized = np.hstack((first_idx, second_idx)).flatten()
        if n_models == 3:
            third_idx = second_idx + n_temporal_nodes * n_spatial_nodes
            perm_vectorized = np.hstack((first_idx, second_idx, third_idx)).flatten()

        return perm_vectorized