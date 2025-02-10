# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import sp, NDArray
from pyinla.configs.submodels_config import BrainiacSubModelConfig
from pyinla.core.submodel import SubModel

import numpy as np
from scipy.sparse import diags, spmatrix
from pyinla import sp, xp


class BrainiacSubModel(SubModel):
    """Fit a regression model."""

    def __init__(
        self,
        config: BrainiacSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        # Load spatial_matrices
        # load covariates Z
        z: NDArray = np.load(self.input_path.joinpath("z.npz"))

        # load projection matrix A
        a: NDArray = np.load(self.input_path.joinpath("a.npz"))

        # TODO: check what if this can be simplified for GPU case
        if xp == np:
            self.z: NDArray = z
            self.a: NDArray = a
        else:
            self.z: NDArray = z
            self.a: NDArray = a

        self._check_dimensions_matrices()

    def _check_dimensions_matrices(self) -> None:
        """Check the dimensions of the model."""

        # check that number of columns in Z matches length of alpha
        # check that number of rows in Z matches number of columns in A
        assert (
            self.z.shape[0] == self.a.shape[1]
        ), "numbers rows in Z must match number of columns in A."

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        alpha = kwargs.get("alpha")
        h = kwargs.get("h")

        # \Phi = 1 / \sum_k=1^B exp(Z^k \alpha) * diag(exp(Z_1 \alpha), exp(Z_2 \alpha), ... )
        exp_Z_alpha = np.exp(self.a @ alpha)
        # print(exp_Z_alpha)
        sum_exp_Z_alpha = np.sum(exp_Z_alpha)
        # print(sum_exp_Z_alpha)

        normalized_exp_Z_alpha = exp_Z_alpha / sum_exp_Z_alpha
        # print(normalized_exp_Z_alpha)

        h2_phi = h**2 * normalized_exp_Z_alpha.flatten()
        Qprior = diags(1 / h2_phi)

        return self.Q_prior
