# Copyright 2024-2025 pyINLA authors. All rights reserved.

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
        self.submodel_type = config.type

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
    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""
        ...
        
    def load_a_predict(self) -> sp.sparse.csc_matrix:
        """Load the design matrix for prediction."""
        a_predict: sp.sparse.csc_matrix = csc_matrix(
            load_npz(self.input_path.joinpath("apr.npz"))
        )
        
        if xp == np:
            self.a_predict: sp.sparse.spmatrix = a_predict
        else:
            self.a_predict: sp.sparse.spmatrix = sp.sparse.csc_matrix(a_predict)
        
        # check that number of columns is the same as in a
        if self.a_predict.shape[1] != self.a.shape[1]:
            raise ValueError(
                f"Number of columns in a_predict ({self.a_predict.shape[1]}) "
                f"does not match number of columns in a ({self.a.shape[1]})."
            )
        
        return self.a_predict

        
