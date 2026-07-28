# Copyright 2024-2025 DALIA authors. All rights reserved.

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, spmatrix

from dalia import NDArray, sp, xp
from dalia.configs.submodels_config import SubModelConfig


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
        self.n_replicates = config.n_replicates

        # --- Load design matrix

        try:
            a: spmatrix = load_npz(self.input_path.joinpath("a.npz"))
            self.a = sp.sparse.csc_matrix(a)
            
            # replicate if n_replicates > 1
            if self.n_replicates > 1:
                self.a = sp.sparse.block_diag([self.a] * self.n_replicates, format="csc")
                
        except FileNotFoundError:
            # check if dense a matrix exists
            try:
                a: NDArray = np.load(self.input_path.joinpath("a.npy"))
                if xp == np:
                    self.a: NDArray = a
                else:
                    self.a: NDArray = xp.array(a)
                    
                if self.n_replicates > 1:
                    self.a = sp.block_diag([self.a] * self.n_replicates)
                    
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No design matrix found under {self.input_path}. Please provide a valid design matrix."
                )

        self.n_latent_parameters: int = self.a.shape[1]
        self.n_latent_parameters_core: int = self.n_latent_parameters // self.n_replicates

        # --- Load latent parameters vector
        try:
            x_initial: NDArray = np.load(self.input_path.joinpath("x.npy"))
            if xp == np:
                self.x_initial: NDArray = x_initial
            else:
                self.x_initial: NDArray = xp.array(x_initial)
                
            # replicate x if needed
            if (
                self.x_initial.shape[0] == self.n_latent_parameters_core
            ) and self.n_replicates > 1:
                self.x_initial = xp.tile(self.x_initial, self.n_replicates)
            
            if self.x_initial.shape[0] != self.n_latent_parameters:
                raise ValueError(
                    f"Length of x_initial ({self.x_initial.shape[0]}) does not match number of latent parameters ({self.n_latent_parameters})."
                )
                    
        except FileNotFoundError:
            self.x_initial: NDArray = xp.zeros((self.a.shape[1]), dtype=float)            

    @abstractmethod
    def _construct_Q_prior_core(self, **kwargs):
        """Construct the prior precision matrix."""
        ... 
        
    def construct_Q_prior(self, **kwargs):
        """Construct the prior precision matrix for the submodel."""
        # defaults to single Q_prior
        if self.n_replicates == 1:
            return self._construct_Q_prior_core(**kwargs)
        # when there are multple replicates
        else:     
            q_prior_core = self._construct_Q_prior_core(**kwargs)
            if sp.sparse.issparse(q_prior_core):
                return sp.sparse.block_diag(
                    [q_prior_core] * self.n_replicates, format="coo"
                )
            return sp.block_diag([q_prior_core] * self.n_replicates)

    def load_a_predict(self) -> sp.sparse.csc_matrix:
        """Load the design matrix for prediction."""
        self.a_predict: sp.sparse.csc_matrix = sp.sparse.csc_matrix(
            load_npz(self.input_path.joinpath("apr.npz"))
        )
        
        if self.n_replicates > 1:
            self.a_predict = sp.sparse.block_diag([self.a_predict] * self.n_replicates, format="csc")

        # check that number of columns is the same as in a
        if self.a_predict.shape[1] != self.a.shape[1]:
            raise ValueError(
                f"Number of columns in a_predict ({self.a_predict.shape[1]}) "
                f"does not match number of columns in a ({self.a.shape[1]})."
            )

        return self.a_predict
