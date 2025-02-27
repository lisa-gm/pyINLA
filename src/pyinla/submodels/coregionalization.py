# Copyright 2024-2025 pyINLA authors. All rights reserved.

import math

import numpy as np
from scipy.sparse import csc_matrix, load_npz, spmatrix

from pyinla import sp, xp
from pyinla.configs.submodels_config import CoregionalizationSubModelConfig
from pyinla.core.submodel import SubModel
from pyinla.submodels.spatial import SpatialSubModel
from pyinla.submodels.spatio_temporal import SpatioTemporalSubModel


class CoregionalizationSubModel(SubModel):
    """Fit a spatio-temporal model."""

    def __init__(
        self,
        config: CoregionalizationSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        self.num_vars = config.num_vars
        self.submodel_type = config.submodel_type

        self.submodels = []
        if config.model_type == "spatial":
            for i in range(self.num_vars):
                self.submodels.append(SpatialSubModel(config.submodel_config))

            self.submodels.append(SpatialSubModel(config.submodel_config_1))
            self.submodels.append(SpatialSubModel(config.submodel_config_2))
        elif config.model_type == "spatio-temporal":
            self.submodels.append(SpatioTemporalSubModel(config.submodel_config_1))
            self.submodels.append(SpatioTemporalSubModel(config.submodel_config_2))
        else:
            raise ValueError(
                "Invalid model type. Must be 'spatial' or 'spatio-temporal'."
            )

        # need to instantiate submodels

    def construct_Q_prior(self, **kwargs):
        """Construct the prior precision matrix."""

        # get theta parameters

        Q_list = [None for _ in range(self.num_vars)]

        for i in range(self.num_vars):
            # TODO: how to pass the correct theta parameters
            # also keeping in mind that sigma_variation always fixed at 1 / log(1) for submodels
            Q_list[i] = self.submodels[i].construct_Q_prior()

        # TODO: don't explicitly construct this but directly pass entries to permutation
        if self.num_vars == 2:
            Q = sp.sparse.vstack(
                [
                    sp.sparse.hstack(
                        [
                            (1 / sigma1**2) * Q_list[0]
                            + (lambda1**2 / sigma2**2) * Q_list[1],
                            (-lambda1 / sigma2**2) * Q2,
                        ]
                    ),
                    sp.sparse.hstack(
                        [
                            (-lambda1 / sigma2**2) * Q_list[1],
                            (1 / sigma2**2) * Q_list[1],
                        ]
                    ),
                ]
            )

        elif self.num_vars == 3:
            Q = sp.sparse.vstack(
                [
                    sp.sparse.hstack(
                        [
                            (1 / sigma1**2) * Q_list[0]
                            + (lambda1**2 / sigma2**2) * Q_list[1]
                            + (lambda3**2 / sigma3**2) * Q_list[2],
                            (-lambda1 / sigma2**2) * Q_list[1]
                            + (lambda2 * lambda3 / sigma3**2) * Q_list[2],
                            -lambda3 / sigma3**2 * Q_list[2],
                        ]
                    ),
                    sp.sparse.hstack(
                        [
                            (-lambda1 / sigma2**2) * Q_list[1]
                            + (lambda2 * lambda3 / sigma3**2) * Q_list[2],
                            (1 / sigma2**2) * Q_list[1]
                            + (lambda2**2 / sigma3**2) * Q_list[2],
                            -lambda2 / sigma3**2 * Q_list[2],
                        ]
                    ),
                    sp.sparse.hstack(
                        [
                            -lambda3 / sigma3**2 * Q_list[2],
                            -lambda2 / sigma3**2 * Q_list[2],
                            (1 / sigma3**2) * Q_list[2],
                        ]
                    ),
                ]
            )

        if self.submodel_type == "spatio_temporal":
            # permute matrix
            permM = self.generate_block_permutation_matrix(
                self.num_vars, self.submodels[0].ns, self.submodels[0].nt
            )
            Q = permM.T @ Q @ permM

            # alternatively
            # perm_vec = self.generate_permutation_indices(
            #     self.num_vars, self.submodels[0].ns, self.submodels[0].nt
            # )
            # Q = Q[perm_vec, :][:, perm_vec]

        return Q

    def generate_block_permutation_matrix(self, n_blocks, block_size, num_vars):
        """
        Generate a permutation matrix using the permutation scheme [0, n, 1, n+1, 2, n+2, ...].
        """
        perm = self.generate_permutation(n_blocks, num_vars)

        n_perm_mat = len(perm) * block_size
        permutation_matrix = sp.csc_matrix((n_perm_mat, n_perm_mat), dtype=int)

        print("permutation vector:")
        for i in range(len(perm)):
            permutation_matrix[
                i * block_size : (i + 1) * block_size,
                perm[i] * block_size : (perm[i] + 1) * block_size,
            ] = sp.eye(block_size)
            seq = list(range(perm[i] * block_size, (perm[i] + 1) * block_size))

            print(seq)

        return permutation_matrix

    def generate_permutation_indices(self, n_blocks, block_size, num_vars):
        """
        Generate a permutation vector containing indices in the pattern:
        [0:block_size, n*block_size:(n+1)*block_size, 1*block_size:(1+1)*block_size, (n+1)*block_size:(n+1+1)*block_size, ...]

        Parameters
        ----------
        n_blocks : int
            Number of blocks.
        block_size : int
            Size of each block.

        Returns
        -------
        np.ndarray
            The generated permutation vector.
        """
        indices = np.arange(n_blocks * block_size)

        first_idx = indices.reshape(n_blocks, block_size)
        second_idx = first_idx + n_blocks * block_size

        if num_vars == 2:
            perm_vectorized = np.hstack((first_idx, second_idx)).flatten()
        if num_vars == 3:
            third_idx = second_idx + n_blocks * block_size
            perm_vectorized = np.hstack((first_idx, second_idx, third_idx)).flatten()

        return perm_vectorized

    def generate_permutation(self, n, num_vars):
        """
        Generate a 1D array in the form [0, n, 2*n, 1, n+1, 2*n+1, 2, n+2, ...].
        """

        first_idx = np.arange(n)
        second_idx = np.arange(n, 2 * n)
        if num_vars == 2:
            perm = np.empty((n + n,), dtype=int)
            perm[0::2] = first_idx
            perm[1::2] = second_idx
        if num_vars == 3:
            third_idx = np.arange(2 * n, 3 * n)
            perm = np.empty((n + n + n,), dtype=int)
            perm[0::3] = first_idx
            perm[1::3] = second_idx
            perm[2::3] = third_idx

        return perm

    def __str__(self) -> str:
        """String representation of the submodel."""
        return (
            " --- CoregionalizationSubModel ---\n"
            f"  num_vars:      {self.num_vars}\n"
            f"  submodel type: {self.submodel_type}\n"
        )
