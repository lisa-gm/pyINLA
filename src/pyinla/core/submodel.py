# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.sparse import spmatrix, csc_matrix, load_npz
from pyinla import xp, sp, NDArray

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
        a: spmatrix = csc_matrix(
            load_npz(Path.joinpath(simulation_path, submodel_config.inputs, "a.npz"))
        )
        if xp == np:
            self.a: sp.sparse.spmatrix = a
        else:
            self.a: sp.sparse.spmatrix = sp.sparse.csc_matrix(a)
        self.n_latent_parameters: int = self.a.shape[1]

        # --- Load latent parameters vector
        try:
            x_initial: NDArray = np.load(
                Path.joinpath(simulation_path, submodel_config.inputs, "x.npy")
            )
            if xp == np:
                self.x_initial: NDArray = x_initial
            else:
                self.x_initial: NDArray = xp.array(x_initial)
        except FileNotFoundError:
            self.x_initial: NDArray = xp.zeros((self.a.shape[1]), dtype=float)

    @abstractmethod
    def construct_Q_prior(self, **kwargs) -> spmatrix:
        """Construct the prior precision matrix."""
        ...
