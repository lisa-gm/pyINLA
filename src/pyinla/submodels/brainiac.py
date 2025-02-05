# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import sp, NDArray
from pyinla.configs.submodels_config import BrainiacSubModelConfig
from pyinla.core.submodel import SubModel

import numpy as np
from scipy.sparse import diags


class BrainiacSubModel(SubModel):
    """Fit a regression model."""

    def __init__(
        self,
        config: BrainiacSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        self.n_fixed_effects: int = config.n_fixed_effects

        # Check that design_matrix shape match number of fixed effects
        assert (
            self.n_fixed_effects == self.n_latent_parameters
        ), f"Design matrix has {self.n_latent_parameters} columns, but expected {self.n_fixed_effects} columns."

        # Load spatial_matrices
        # load covariates Z
        z: NDArray = np.load(self.input_path.joinpath("z.npz"))

        # load projection matrix A
        a: NDArray = np.load(self.input_path.joinpath("a.npz"))

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        alpha = kwargs.get("alpha")
        h2 = kwargs.get("h2")

        # \Phi = 1 / \sum_k=1^B exp(Z^k \alpha) * diag(exp(Z_1 \alpha), exp(Z_2 \alpha), ... )
        exp_Z_alpha = np.exp(self.a @ alpha)
        # print(exp_Z_alpha)
        sum_exp_Z_alpha = np.sum(exp_Z_alpha)
        # print(sum_exp_Z_alpha)

        normalized_exp_Z_alpha = exp_Z_alpha / sum_exp_Z_alpha
        # print(normalized_exp_Z_alpha)

        h2_phi = h2 * normalized_exp_Z_alpha.flatten()
        Qprior = diags(1 / h2_phi)

        return self.Q_prior
