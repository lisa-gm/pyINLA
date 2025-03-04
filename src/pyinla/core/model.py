# Copyright 2024-2025 pyINLA authors. All rights reserved.

import os
from abc import ABC
from pathlib import Path

import numpy as np
from scipy.sparse import spmatrix

from pyinla import ArrayLike, NDArray, sp, xp
from pyinla.configs.likelihood_config import LikelihoodConfig
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
            if isinstance(submodel, SpatioTemporalSubModel):
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
            elif isinstance(submodel, SpatialSubModel):
                # spatial range
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

                # spatial variation
                if isinstance(submodel.config.ph_e, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_e,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_e, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_e,
                            hyperparameter_type="sigma_e",
                        )
                    )
            elif isinstance(submodel, RegressionSubModel):
                ...

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
        self.latent_parameters_idx: list[int] = [0]

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

        self.n_observations: int = self.y.shape[0]

        # --- Initialize likelihood
        # TODO: clean this -> so that for brainiac model we don't add additional hyperperameter
        if likelihood_config.type == "gaussian":
            self.likelihood: Likelihood = GaussianLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )

            # Instantiate the prior hyperparameters for the likelihood
            if isinstance(
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
                if isinstance(submodel, SpatioTemporalSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])
                elif isinstance(submodel, SpatialSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])
                elif isinstance(submodel, RegressionSubModel):
                    ...

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
                elif isinstance(submodel, SpatialSubModel):
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

        if self.likelihood_config.type == "gaussian":
            kwargs = {
                "eta": eta,
                "theta": float(self.theta[-1]),
            }
        else:
            kwargs = {
                "eta": eta,
            }

        d_matrix = self.likelihood.evaluate_hessian_likelihood(**kwargs)
        # print("dim d_matrix: ", d_matrix.shape)
        # print("d_matrix: \n", d_matrix.toarray()[:5, :5])
        # print("dim(a): ", self.a.shape)
        # print("A: ", self.a.toarray()[:5, :5])
        # print("dim(Q_prior): ", self.Q_prior.shape)
        # print("Q_prior: ", self.Q_prior[:5, :5])

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
            # print("tmp: ", tmp)
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

    def get_solver_parameters(self) -> dict:
        """Get the solver parameters."""
        diagonal_blocksize = None
        n_diag_blocks = None
        arrowhead_blocksize = 0
        if isinstance(self.submodels[0], SpatioTemporalSubModel):
                diagonal_blocksize = self.submodels[0].ns
                n_diag_blocks = self.submodels[0].nt

        for i in range(1, len(self.submodels)):
                if isinstance(self.submodels[i], RegressionSubModel):
                    arrowhead_blocksize += self.submodels[i].n_latent_parameters

        param = {
            "diagonal_blocksize": diagonal_blocksize,
            "n_diag_blocks": n_diag_blocks,
            "arrowhead_blocksize": arrowhead_blocksize,
        }

        return param