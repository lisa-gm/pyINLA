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

# from pyinla.utils import bdiag_tilling

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
        assert self.n_models == len(self.models), "Number of models does not match the number of models in the CoregionalModelConfig"
        self.n_spatial_nodes: int = None
        self.n_temporal_nodes: int = 1
        self.n_fixed_effects_per_model: int = 0
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
                if len(model.submodels) > 1:
                    if isinstance(model.submodels[1], RegressionSubModel):
                        self.n_fixed_effects_per_model = model.submodels[1].n_fixed_effects
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
                else:
                    raise ValueError(
                        "Invalid model type. Must be 'spatial' or 'spatio-temporal'."
                    )
                if len(model.submodels) > 1:
                    if isinstance(model.submodels[1], RegressionSubModel):
                        if model.submodels[1].n_fixed_effects != self.n_fixed_effects_per_model:
                            raise ValueError(
                                f"Model {model} has a different number of fixed effects than the first model"
                            )

        # Get Models() hyperparameters
        theta: ArrayLike = []
        theta_keys: ArrayLike = []
        self.hyperparameters_idx: ArrayLike = [0]
        self.prior_hyperparameters: list[PriorHyperparameters] = []

        for model in self.models:
            theta_model = model.theta
            theta_keys_model = model.theta_keys

            # remove the theta that correspond to the "sigma_xx" where x can be whatever
            sigma_indices = [i for i, key in enumerate(theta_keys_model) if re.match(r"sigma_\w+", key)]
            theta_model = [theta for i, theta in enumerate(theta_model) if i not in sigma_indices]
            theta_keys_model = [key for i, key in enumerate(theta_keys_model) if i not in sigma_indices]

            theta.append(xp.array(theta_model))
            theta_keys += theta_keys_model

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_model)
            )

            # Get the prior hyperparameters of the model
            self.prior_hyperparameters += [prior_hyperparameters for i, prior_hyperparameters in enumerate(model.prior_hyperparameters) if i not in sigma_indices]

        # Initialize the Coregional Hyperparameters:
        theta_coregional_model, theta_keys_coregional_model = coregional_model_config.read_hyperparameters()
        theta.append(xp.array(theta_coregional_model))
        theta_keys += theta_keys_coregional_model

        self.hyperparameters_idx.append(
            self.hyperparameters_idx[-1] + len(theta_coregional_model)
        )

        # Finalize the hyperparameters
        self.theta: NDArray = xp.concatenate(theta)
        self.n_hyperparameters = self.theta.size
        self.theta_keys: NDArray = theta_keys

        # Initialize the Coregional Prior Hyperparameters
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
        ).tocsc()

        # self.a: spmatrix = sp.sparse.vstack(
        #     [
        #         sp.sparse.hstack(
        #             [
        #                 self.models[0].submodels[0].a,
        #                 sp.sparse.csc_matrix(
        #                     (
        #                         self.models[0].n_observations,
        #                         self.models[1].n_latent_parameters
        #                         - self.n_fixed_effects_per_model,
        #                     )
        #                 ),
        #                 self.models[0].submodels[1].a,
        #                 sp.sparse.csc_matrix(
        #                     (
        #                         self.models[0].n_observations,
        #                         self.n_fixed_effects_per_model,
        #                     )
        #                 ),
        #             ]
        #         ),
        #         sp.sparse.hstack(
        #             [
        #                 sp.sparse.csc_matrix(
        #                     (
        #                         self.models[1].n_observations,
        #                         self.models[0].n_latent_parameters
        #                         - self.n_fixed_effects_per_model,
        #                     )
        #                 ),
        #                 self.models[1].submodels[0].a,
        #                 sp.sparse.csc_matrix(
        #                     (
        #                         self.models[1].n_observations,
        #                         self.n_fixed_effects_per_model,
        #                     )
        #                 ),
        #                 self.models[1].submodels[1].a,
        #             ]
        #         ),
        #     ]
        # )

        # permutation_latent_variables = self._generate_permutation_indices_for_a_new(
        #     self.n_temporal_nodes,
        #     self.n_spatial_nodes,
        #     self.n_models,
        #     self.n_fixed_effects_per_model,
        # )

        permutation_latent_variables = self._generate_permutation_indices_for_a(
            self.n_temporal_nodes,
            self.n_spatial_nodes,
            self.n_models,
            self.n_fixed_effects_per_model,
        )

        self.a = self.a[:, permutation_latent_variables]
        self.x = self.x[permutation_latent_variables]

        # --- Recurrent variables
        self.Q_prior = None
        self.Q_prior_data_mapping = [0]
        self.Q_conditional = None
        self.Q_conditional_data_mapping = [0]

    def construct_Q_prior(self) -> spmatrix:
        Qst_list: list = []
        Q_r: list = []
        for i, model in enumerate(self.models):
            submodel_st = model.submodels[0]
            # Get the spatio-temporal submodel idx
            kwargs_st = {}
            for hp_idx in range(
                model.hyperparameters_idx[0], model.hyperparameters_idx[1]
            ):
                kwargs_st[model.theta_keys[hp_idx]] = float(model.theta[hp_idx])
            print("kwargs_st: ", kwargs_st)
            Qst_list.append(submodel_st.construct_Q_prior(**kwargs_st).tocsc())

            if len(model.submodels) > 1:
                # Create the regression tip
                submodel_r = model.submodels[1]
                # Get the spatio-temporal submodel idx
                kwargs_r = {}
                for hp_idx in range(
                    model.hyperparameters_idx[1], model.hyperparameters_idx[2]
                ):
                    kwargs_r[model.theta_keys[hp_idx]] = float(model.theta[hp_idx])
                Q_r.append(submodel_r.construct_Q_prior(**kwargs_r).tocsc())

        # print(self.theta_keys.index("sigma_0"))
        # print(self.theta[self.theta_keys.index("sigma_0")])

        sigma_0 = xp.exp(self.theta[self.theta_keys.index("sigma_0")])
        sigma_1 = xp.exp(self.theta[self.theta_keys.index("sigma_1")])

        lambda_0_1 = self.theta[self.theta_keys.index("lambda_0_1")]

        if self.n_models == 2:
            Qprior_st = sp.sparse.vstack(
                [
                    sp.sparse.hstack(
                        [
                            (1 / sigma_0**2) * Qst_list[0]
                            + (lambda_0_1**2 / sigma_1**2) * Qst_list[1],
                            (-lambda_0_1 / sigma_1**2) * Qst_list[1],
                        ]
                    ),
                    sp.sparse.hstack(
                        [
                            (-lambda_0_1 / sigma_1**2) * Qst_list[1],
                            (1 / sigma_1**2) * Qst_list[1],
                        ]
                    ),
                ]
            ).tocsc()
        elif self.n_models == 3:
            sigma_2 = xp.exp(self.theta[self.theta_keys.index("sigma_2")])

            lambda_0_2 = self.theta[self.theta_keys.index("lambda_0_2")]
            lambda_1_2 = self.theta[self.theta_keys.index("lambda_1_2")]

            Qprior_st = sp.sparse.vstack(
                [
                    sp.sparse.hstack(
                        [
                            (1 / sigma_0**2) * Qst_list[0]
                            + (lambda_0_1**2 / sigma_1**2) * Qst_list[1]
                            + (lambda_1_2**2 / sigma_2**2) * Qst_list[2],
                            (-lambda_0_1 / sigma_1**2) * Qst_list[1]
                            + (lambda_0_2 * lambda_1_2 / sigma_2**2) * Qst_list[2],
                            -lambda_1_2 / sigma_2**2 * Qst_list[2],
                        ]
                    ),
                    sp.sparse.hstack(
                        [
                            (-lambda_0_1 / sigma_1**2) * Qst_list[1]
                            + (lambda_0_2 * lambda_1_2 / sigma_2**2) * Qst_list[2],
                            (1 / sigma_1**2) * Qst_list[1]
                            + (lambda_0_2**2 / sigma_2**2) * Qst_list[2],
                            -lambda_0_2 / sigma_2**2 * Qst_list[2],
                        ]
                    ),
                    sp.sparse.hstack(
                        [
                            -lambda_1_2 / sigma_2**2 * Qst_list[2],
                            -lambda_0_2 / sigma_2**2 * Qst_list[2],
                            (1 / sigma_2**2) * Qst_list[2],
                        ]
                    ),
                ]
            ).tocsc()

        # Apply the permutation to the Qprior_st
        if self.coregionalization_type == "spatio_temporal":
            # Permute matrix
            p_vec = self._generate_permutation_indices(self.n_temporal_nodes, self.n_spatial_nodes, self.n_models)
            # p_vec = np.arange(
            #     self.n_models * (self.n_temporal_nodes * self.n_spatial_nodes)
            # )
            # permute Q
            Qprior_st_perm = Qprior_st[p_vec, :][:, p_vec]

        if Q_r != []:
            # if Q_r:
            # Qprior_reg = sp.sparse.bmat([[Q] for Q in Q_r]).tocsc()
            # self.Q_prior = sp.sparse.bmat([[Qprior_st_perm, None], [None, Qprior_reg]]).tocsc()

            # Qprior_reg = bdiag_tilling(Q_r)
            # self.Q_prior = bdiag_tilling([Qprior_st_perm, Qprior_reg])

            Qprior_reg = sp.sparse.block_diag(Q_r).tocsc()
            print("Qprior_reg.shape: ", Qprior_reg.shape)
            self.Q_prior = sp.sparse.block_diag([Qprior_st_perm, Qprior_reg]).tocsc()
        else:
            self.Q_prior = Qprior_st_perm

        return self.Q_prior

    # def construct_Q_prior(self) -> spmatrix:

    #     Qst_list: list = []
    #     Q_r: list = []
    #     for i, model in enumerate(self.models):
    #         submodel_st = model.submodels[0]
    #         # Get the spatio-temporal submodel idx
    #         kwargs_st = {}
    #         for hp_idx in range(
    #             model.hyperparameters_idx[0], model.hyperparameters_idx[1]
    #         ):
    #             kwargs_st[model.theta_keys[hp_idx]] = float(model.theta[hp_idx])
    #         print("kwargs_st: ", kwargs_st)
    #         Qst_list.append(submodel_st.construct_Q_prior(**kwargs_st).tocsc())

    #         if len(model.submodels) > 1:
    #             # Create the regression tip
    #             submodel_r = model.submodels[1]
    #             # Get the spatio-temporal submodel idx
    #             kwargs_r = {}
    #             for hp_idx in range(
    #                 model.hyperparameters_idx[1], model.hyperparameters_idx[2]
    #             ):
    #                 kwargs_r[model.theta_keys[hp_idx]] = float(model.theta[hp_idx])
    #             Q_r.append(submodel_r.construct_Q_prior(**kwargs_r).tocsc())

    #     # print(self.theta_keys.index("sigma_0"))
    #     # print(self.theta[self.theta_keys.index("sigma_0")])

    #     sigma_0 = xp.exp(self.theta[self.theta_keys.index("sigma_0")])
    #     sigma_1 = xp.exp(self.theta[self.theta_keys.index("sigma_1")])
    #     print("sigma_0: ", sigma_0)
    #     print("sigma_1: ", sigma_1)
    #     print("Q1[:5, :5]: \n", Qst_list[0][:5, :5].toarray())
    #     print("Q2[:5, :5]: \n", Qst_list[1][:5, :5].toarray())

    #     lambda_0_1 = self.theta[self.theta_keys.index("lambda_0_1")]

    #     if self.n_models == 2:
    #         Qprior_st = sp.sparse.vstack(
    #             [
    #                 sp.sparse.hstack(
    #                     [
    #                         (1 / sigma_0**2) * Qst_list[0]
    #                         + (lambda_0_1**2 / sigma_1**2) * Qst_list[1],
    #                         (-lambda_0_1 / sigma_1**2) * Qst_list[1],
    #                     ]
    #                 ),
    #                 sp.sparse.hstack(
    #                     [
    #                         (-lambda_0_1 / sigma_1**2) * Qst_list[1],
    #                         (1 / sigma_1**2) * Qst_list[1],
    #                     ]
    #                 ),
    #             ]
    #         ).tocsc()
    #     elif self.n_models == 3:
    #         sigma_2 = xp.exp(self.theta[self.theta_keys.index("sigma_2")])

    #         lambda_0_2 = self.theta[self.theta_keys.index("lambda_0_2")]
    #         lambda_1_2 = self.theta[self.theta_keys.index("lambda_1_2")]

    #         Qprior_st = sp.sparse.vstack(
    #             [
    #                 sp.sparse.hstack(
    #                     [
    #                         (1 / sigma_0**2) * Qst_list[0]
    #                         + (lambda_0_1**2 / sigma_1**2) * Qst_list[1]
    #                         + (lambda_1_2**2 / sigma_2**2) * Qst_list[2],
    #                         (-lambda_0_1 / sigma_1**2) * Qst_list[1]
    #                         + (lambda_0_2 * lambda_1_2 / sigma_2**2) * Qst_list[2],
    #                         -lambda_1_2 / sigma_2**2 * Qst_list[2],
    #                     ]
    #                 ),
    #                 sp.sparse.hstack(
    #                     [
    #                         (-lambda_0_1 / sigma_1**2) * Qst_list[1]
    #                         + (lambda_0_2 * lambda_1_2 / sigma_2**2) * Qst_list[2],
    #                         (1 / sigma_1**2) * Qst_list[1]
    #                         + (lambda_0_2**2 / sigma_2**2) * Qst_list[2],
    #                         -lambda_0_2 / sigma_2**2 * Qst_list[2],
    #                     ]
    #                 ),
    #                 sp.sparse.hstack(
    #                     [
    #                         -lambda_1_2 / sigma_2**2 * Qst_list[2],
    #                         -lambda_0_2 / sigma_2**2 * Qst_list[2],
    #                         (1 / sigma_2**2) * Qst_list[2],
    #                     ]
    #                 ),
    #             ]
    #         ).tocsc()

    #     # Apply the permutation to the Qprior_st
    #     if self.coregionalization_type == "spatio_temporal":
    #         # Permute matrix
    #         # p_vec = self._generate_permutation_indices(self.n_temporal_nodes, self.n_spatial_nodes, self.n_models)
    #         p_vec = np.arange(
    #             self.n_models * (self.n_temporal_nodes * self.n_spatial_nodes)
    #         )
    #         # permute Q
    #         Qprior_st_perm = Qprior_st[p_vec, :][:, p_vec]

    #     if Q_r != []:
    #         # if Q_r:
    #         # Qprior_reg = sp.sparse.bmat([[Q] for Q in Q_r]).tocsc()
    #         # self.Q_prior = sp.sparse.bmat([[Qprior_st_perm, None], [None, Qprior_reg]]).tocsc()

    #         Qprior_reg = sp.sparse.block_diag(Q_r).tocsc()
    #         print("Qprior_reg.shape: ", Qprior_reg.shape)
    #         self.Q_prior = sp.sparse.block_diag([Qprior_st_perm, Qprior_reg]).tocsc()
    #     else:
    #         self.Q_prior = Qprior_st_perm

    #     return self.Q_prior

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

        d_list = []
        for i, model in enumerate(self.models):
            if model.likelihood_config.type == "gaussian":
                kwargs = {
                    "eta": eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                    "theta": float(self.theta[self.hyperparameters_idx[i+1]-1]),
                }
            else:
                kwargs = {
                    "eta": eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                }

            d_list.append(model.likelihood.evaluate_hessian_likelihood(**kwargs))

        d_matrix = sp.sparse.block_diag(d_list).tocsc()
        # tmp = self.a.T @ d_matrix @ self.a
        # perm_vec = self._generate_permutation_indices_for_a_new(
        #     self.n_temporal_nodes,
        #     self.n_spatial_nodes,
        #     self.n_models,
        #     self.n_fixed_effects_per_model,
        # )
        # tmp = tmp[perm_vec, :][:, perm_vec]
        # self.Q_conditional = self.Q_prior - tmp

        # print("here...")
        self.Q_conditional = self.Q_prior - self.a.T @ d_matrix @ self.a

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

        gradient_vector_list = []
        for i, model in enumerate(self.models):

            gradient_likelihood = model.likelihood.evaluate_gradient_likelihood(
                eta=eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                y=self.y[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                theta=float(self.theta[self.hyperparameters_idx[i+1]-1]),
            )

            gradient_vector_list.append(gradient_likelihood)

        gradient_likelihood = xp.concatenate(gradient_vector_list)

        information_vector: NDArray = (
            -1 * self.Q_prior @ x_i + self.a.T @ gradient_likelihood
        )

        return information_vector

    def evaluate_likelihood(self, 
            eta: NDArray,
        ) -> float:

        likelihood: float = 0.
        for i, model in enumerate(self.models):

            likelihood += model.likelihood.evaluate_likelihood(
                eta=eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                y=self.y[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                theta=float(self.theta[self.hyperparameters_idx[i+1]-1]),
            )

        return likelihood

    def evaluate_log_prior_hyperparameters(self) -> float:
        """Evaluate the log prior hyperparameters."""
        log_prior = 0.0

        theta_interpret = self.theta

        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            log_prior += prior_hyperparameter.evaluate_log_prior(theta_interpret[i])

        return log_prior

    def __str__(self) -> str:
        """String representation of the model."""
        # Collect general information about the model
        coregional_model_info = [
            " --- Coregional Model ---",
            f"n_hyperparameters: {self.n_hyperparameters}",
            f"n_latent_parameters: {self.n_latent_parameters}",
            f"n_observations: {self.n_observations}",
        ]

        # Collect each submodel's information
        model_info = [str(model) for model in self.models]

        # Combine model information and submodel information
        return "\n".join(coregional_model_info + model_info)

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

    def _generate_permutation_indices_for_a_new(
        self,
        n_temporal_nodes: int,
        n_spatial_nodes: int,
        n_models: int,
        n_fixed_effects_per_model: int,
    ):
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
        # indices_fixed_effects_1 = len(indices) + np.arange(n_fixed_effects_per_model)
        print("len(indices): ", len(indices))
        print("n_fixed_effects_per_model: ", n_fixed_effects_per_model)

        first_idx = indices.reshape(n_temporal_nodes, n_spatial_nodes)
        second_idx = first_idx + n_temporal_nodes * n_spatial_nodes
        indices_fixed_effects_1 = 2 * n_temporal_nodes * n_spatial_nodes + np.arange(
            n_fixed_effects_per_model
        )
        indices_fixed_effects_2 = n_fixed_effects_per_model + indices_fixed_effects_1
        print("indices_fixed_effects_2: ", indices_fixed_effects_2)

        if n_models == 2:
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                ]
            )
        elif n_models == 3:
            third_idx = (
                second_idx
                + n_temporal_nodes * n_spatial_nodes
                + 2 * n_fixed_effects_per_model
            )
            indices_fixed_effects_3 = (
                3 * n_temporal_nodes * n_spatial_nodes
                + 2 * n_fixed_effects_per_model
                + np.arange(n_fixed_effects_per_model)
            )
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx, third_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                    indices_fixed_effects_3,
                ]
            )

        return perm_vectorized

    def _generate_permutation_indices_for_a(self, n_temporal_nodes: int, n_spatial_nodes: int, n_models: int, n_fixed_effects_per_model: int):
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
        indices_fixed_effects_1 = len(indices) + np.arange(n_fixed_effects_per_model)
        print("len(indices): ", len(indices))
        print("n_fixed_effects_per_model: ", n_fixed_effects_per_model)

        first_idx = indices.reshape(n_temporal_nodes, n_spatial_nodes)
        second_idx = first_idx + n_fixed_effects_per_model + n_temporal_nodes * n_spatial_nodes
        indices_fixed_effects_2 = 2*n_temporal_nodes*n_spatial_nodes + n_fixed_effects_per_model + np.arange(n_fixed_effects_per_model)
        print("indices_fixed_effects_2: ", indices_fixed_effects_2)

        if n_models == 2:
            perm_vectorized = np.concatenate([np.hstack((first_idx, second_idx)).flatten(), indices_fixed_effects_1, indices_fixed_effects_2])
        elif n_models == 3:
            third_idx = second_idx + n_temporal_nodes * n_spatial_nodes + n_fixed_effects_per_model
            indices_fixed_effects_3 = 3*n_temporal_nodes*n_spatial_nodes + 2*n_fixed_effects_per_model + np.arange(n_fixed_effects_per_model)
            perm_vectorized = np.concatenate([np.hstack((first_idx, second_idx, third_idx)).flatten(), indices_fixed_effects_1, indices_fixed_effects_2, indices_fixed_effects_3])

        return perm_vectorized

    def get_solver_parameters(self) -> dict:
        """Get the solver parameters."""
        diagonal_blocksize = self.n_models * self.n_spatial_nodes
        n_diag_blocks = self.n_temporal_nodes
        arrowhead_blocksize = self.n_fixed_effects_per_model * self.n_models

        param = {
            "diagonal_blocksize": diagonal_blocksize,
            "n_diag_blocks": n_diag_blocks,
            "arrowhead_blocksize": arrowhead_blocksize,
        }

        return param
