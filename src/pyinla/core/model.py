# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC

from pyinla import ArrayLike, xp
from scipy.sparse import sparray, coo_matrix

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.core.submodel import SubModel

from pyinla.likelihoods.binomial import BinomialLikelihood
from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.likelihoods.poisson import PoissonLikelihood
from pyinla.utils import mapping


class Model(ABC):
    """Core class for statistical models."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the model."""
        self.pyinla_config = pyinla_config

        self.submodels: list[SubModel] = ...

        # Initialize parameters array (theta)
        self.n_hyperparameters = 0
        self.n_latent_parameters = 0
        self.hyperparameters_idx = [0]
        self.latent_parameters_idx = [0]
        for submodel in self.submodels:
            self.n_hyperparameters += submodel.n_hyperparameters
            self.n_latent_parameters += submodel.n_latent_parameters

            self.hyperparameters_idx.append(self.n_hyperparameters)
            self.latent_parameters_idx.append(self.n_latent_parameters)

        self.theta = xp.zeros(self.n_hyperparameters)
        self.x = xp.zeros(self.n_latent_parameters)

        for i, submodel in enumerate(self.submodels):
            self.theta[
                self.hyperparameters_idx[i] : self.hyperparameters_idx[i + 1]
            ] = mapping.dict2array(submodel.theta_initial)

        # --- Load design matrix and latent parameters from the submodels
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
            ] = submodel.x

        self.a = coo_matrix(
            (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
            shape=(submodel.a.shape[0], self.n_latent_parameters),
        )

        # --- Load observation vector
        self.y = xp.load(pyinla_config.input_dir / "y.npy")
        self.n_observations = self.y.shape[0]

        # --- Initialize likelihood
        if self.pyinla_config.model.likelihood.type == "gaussian":
            self.likelihood = GaussianLikelihood(pyinla_config, self.n_observations)
        elif self.pyinla_config.likelihood.type == "poisson":
            self.likelihood = PoissonLikelihood(pyinla_config, self.n_observations)
        elif self.pyinla_config.likelihood.type == "binomial":
            self.likelihood = BinomialLikelihood(pyinla_config, self.n_observations)
