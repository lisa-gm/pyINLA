# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, load_npz, spmatrix

from pyinla import NDArray, sp, xp
from pyinla.configs.submodels_config import SubModelConfig


class SubModel(ABC):
    """Abstract core class for statistical models."""

    def __init__(
        self,
        config: SubModelConfig,
    ) -> None:
        """Initializes the model."""
        self.config = config
        self.input_path = Path(config.input_dir)

        # --- Load design matrix
        a: spmatrix = csc_matrix(load_npz(self.input_path.joinpath("a.npz")))
        if xp == np:
            self.a: sp.sparse.spmatrix = a
        else:
            self.a: sp.sparse.spmatrix = sp.sparse.csc_matrix(a)
        self.n_latent_parameters: int = self.a.shape[1]

        # --- Load latent parameters vector
        try:
            x_initial: NDArray = np.load(self.input_path.joinpath("x.npy"))
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
