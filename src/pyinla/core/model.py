# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC

from pyinla import ArrayLike, xp
from scipy.sparse import sparray, coo_matrix

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.core.submodel import SubModel
from pyinla.submodels.regression import RegressionModel
from pyinla.submodels.spatio_temporal import SpatioTemporalModel

from pyinla.likelihoods.binomial import BinomialLikelihood
from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.likelihoods.poisson import PoissonLikelihood

from pyinla.core.pyinla_config import (
    RegressionSubModelConfig,
    SpatioTemporalSubModelConfig,
)


class Model(ABC):
    """Core class for statistical models."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the model."""
        self.pyinla_config: PyinlaConfig = pyinla_config

        # --- Initialize the submodels
        self.submodels: list[SubModel] = []
        submodel_to_instanciate = [xp.arrange(len(pyinla_config.model.submodels))]
        for i, submodel_config in enumerate(self.pyinla_config.model.submodels):
            if isinstance(submodel_config, SpatioTemporalSubModelConfig):
                self.submodels.append(
                    SpatioTemporalModel(submodel_config, pyinla_config.input_dir)
                )

                submodel_to_instanciate = submodel_to_instanciate[
                    submodel_to_instanciate != i
                ]

        for i in submodel_to_instanciate:
            if isinstance(submodel_config[i], RegressionSubModelConfig):
                self.submodels.append(
                    RegressionModel(submodel_config, pyinla_config.input_dir)
                )
            else:
                raise ValueError("Unknown submodel type.")

        # --- Initialize the hyperparameters array
        theta = []
        theta_keys = []
        self.hyperparameters_idx = [0]

        for submodel_config in self.pyinla_config.model.submodels:
            theta_submodel, theta_keys_submodel = submodel_config.read_hyperparameters()

            theta.append(theta_submodel)
            theta_keys.append(theta_keys_submodel)

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_submodel)
            )

        lh_hyperparameters, lh_hyperparameters_keys = (
            self.pyinla_config.model.likelihood.read_hyperparameters()
        )

        theta.append(lh_hyperparameters)
        theta_keys.append(lh_hyperparameters_keys)

        self.theta = xp.concatenate(theta)
        self.theta_keys = xp.concatenate(theta_keys)

        self.n_hyperparameters = len(self.theta)

        # --- Initialize the latent parameters and the design matrix
        self.n_latent_parameters = 0
        self.latent_parameters_idx = [0]

        for submodel in self.submodels:
            self.n_latent_parameters += submodel.n_latent_parameters
            self.latent_parameters_idx.append(self.n_latent_parameters)

        self.x: ArrayLike = xp.zeros(self.n_latent_parameters)

        data = []
        rows = []
        cols = []
        for i, submodel in enumerate(self.submodels):
            data.append(submodel.a.data)
            rows.append(submodel.a.row)
            cols.append(
                submodel.a.col
                + xp.oneslike(submodel.a.col.size[0]) * self.latent_parameters_idx[i]
            )

            self.x[
                self.latent_parameters_idx[i] : self.latent_parameters_idx[i + 1]
            ] = submodel.x_initial

        self.a: sparray = coo_matrix(
            (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
            shape=(submodel.a.shape[0], self.n_latent_parameters),
        )

        # --- Load observation vector
        self.y = xp.load(pyinla_config.input_dir / "y.npy")
        self.n_observations = self.y.shape[0]

        # --- Initialize likelihood
        if self.pyinla_config.model.likelihood.type == "gaussian":
            self.likelihood = GaussianLikelihood(pyinla_config, self.n_observations)
        elif self.pyinla_config.model.likelihood.type == "poisson":
            self.likelihood = PoissonLikelihood(pyinla_config, self.n_observations)
        elif self.pyinla_config.model.likelihood.type == "binomial":
            self.likelihood = BinomialLikelihood(pyinla_config, self.n_observations)

    def construct_Q_prior(self) -> sparray:

        # Concatenate data, indices, and indptr to form the block diagonal matrix Q
        data = np.concatenate([Q_spatio_temporal_data, Q_fixed_effects_data])
        indices = np.concatenate(
            [
                Q_spatio_temporal_indices,
                Q_fixed_effects_indices + Q_spatio_temporal_shape[1],
            ]
        )
        indptr = np.concatenate(
            [
                Q_spatio_temporal_indptr,
                Q_spatio_temporal_indptr[-1] + Q_fixed_effects_indptr[1:],
            ]
        )

        Q_prior = csc_matrix(
            (data, indices, indptr),
            shape=(
                Q_spatio_temporal.shape[0] + self.nb,
                Q_spatio_temporal.shape[1] + self.nb,
            ),
        )

        self.Q_prior = ...

        return self.Q_prior

    def construct_Q_conditional(
        self,
        Q_prior: sparray,
        hessian_likelihood: sparray,
    ) -> float:
        """Construct the conditional precision matrix.

        Note
        ----
        Input of the hessian of the likelihood is a diagonal matrix.
        The negative hessian is required, therefore the minus in front.

        """

        Q_conditional = self.Q_prior - self.a.T @ hessian_likelihood @ self.a

        return Q_conditional
