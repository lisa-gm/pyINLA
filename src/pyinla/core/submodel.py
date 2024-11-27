# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod
from pathlib import Path

from scipy.sparse import sparray, coo_matrix
from pyinla import ArrayLike, comm_rank, comm_size, sparse, xp

from pyinla.core.pyinla_config import SubModelConfig


class SubModel(ABC):
    """Abstract core class for statistical models."""

    def __init__(
        self,
        submodel_config: SubModelConfig,
        simulation_path: Path,
    ) -> None:
        """Initializes the model."""
        self.submodel_config = submodel_config

        # --- Load design matrix
        self.a = coo_matrix(
            sparse.load_npz(
                Path.joinpath(simulation_path, submodel_config.inputs, "a.npz")
            )
        )
        self.n_latent_parameters = self.a.shape[1]

        # --- Load latent parameters vector
        try:
            self.x = xp.load(
                Path.joinpath(simulation_path, submodel_config.inputs, "x.npy")
            )
        except FileNotFoundError:
            self.x = xp.zeros((self.a.shape[1]), dtype=float)

    @abstractmethod
    def get_hyperparameters(self) -> dict:
        """Get the initial hyperparameters of the model. This dictionary is constructed
        at instanciation of the model. It has to be stored in the model as
        hyperparameters is specific to the model.

        Returns
        -------
        hyperparameters_inital_model : dict
            Dictionary of initial hyperparameters.
        """
        ...

    @abstractmethod
    def construct_Q_prior(self, theta_model: dict = None) -> sparray:
        """Construct the prior precision matrix."""
        pass

    @abstractmethod
    def construct_Q_conditional(
        self,
        Q_prior: sparray,
        a: sparray,
        hessian_likelihood: sparray,
    ) -> sparray:
        """Construct the conditional precision matrix.
        #TODO: hessian_likelihood always diagonal (for all models). How to pass this best?
        """
        pass
