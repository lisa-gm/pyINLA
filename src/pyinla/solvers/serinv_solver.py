# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import sparray

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.core.solver import Solver

try:
    from serinv import pobtaf, pobtas, pobtf
except ImportError:
    print("Serinv not installed. Please install serinv to use SerinvSolver.")


class SerinvSolverCPU(Solver):
    """Serinv Solver class."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        diagonal_blocksize: int,
        arrowhead_blocksize: int,
        n_diagonal_blocks: int,
    ) -> None:
        """Initializes the SerinV solver."""
        super().__init__(pyinla_config)

        self.diagonal_blocksize = diagonal_blocksize
        self.arrowhead_blocksize = arrowhead_blocksize
        self.n_diagonal_blocks = n_diagonal_blocks
        self.dtype = np.float64

        # --- Initialize pinned-memory for BTA-array storage
        self.A_diagonal_blocks = np.empty(
            (self.n_diagonal_blocks, self.diagonal_blocksize, self.diagonal_blocksize),
            dtype=self.dtype,
        )
        self.A_lower_diagonal_blocks = np.empty(
            (
                self.n_diagonal_blocks - 1,
                self.diagonal_blocksize,
                self.diagonal_blocksize,
            ),
            dtype=self.dtype,
        )
        self.A_arrow_bottom_blocks = np.empty(
            (self.n_diagonal_blocks, self.arrowhead_blocksize, self.diagonal_blocksize),
            dtype=self.dtype,
        )
        self.A_arrow_tip_block = np.empty(
            (self.arrowhead_blocksize, self.arrowhead_blocksize),
            dtype=self.dtype,
        )

        # Print the allocated memory for the BTA-array
        total_bytes = (
            self.A_diagonal_blocks.nbytes
            + self.A_lower_diagonal_blocks.nbytes
            + self.A_arrow_bottom_blocks.nbytes
            + self.A_arrow_tip_block.nbytes
        )
        total_gb = total_bytes / (1024**3)
        print(f"Allocated memory for BTA-array: {total_gb:.2f} GB", flush=True)

        # --- Make aliases for L
        self.L_diagonal_blocks = self.A_diagonal_blocks
        self.L_lower_diagonal_blocks = self.A_lower_diagonal_blocks
        self.L_arrow_bottom_blocks = self.A_arrow_bottom_blocks
        self.L_arrow_tip_block = self.A_arrow_tip_block

    def cholesky(self, A: sparray, sparsity: str = "bta") -> None:
        """Compute Cholesky factor of input matrix."""

        if sparsity == "bta":
            self._sparray_to_structured(A, sparsity="bta")

            (
                self.L_diagonal_blocks,
                self.L_lower_diagonal_blocks,
                self.L_arrow_bottom_blocks,
                self.L_arrow_tip_block,
            ) = pobtaf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
            )

        elif sparsity == "bt":
            self._sparray_to_structured(A, sparsity="bt")

            (
                self.L_diagonal_blocks,
                self.L_lower_diagonal_blocks,
            ) = pobtf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
            )

    def solve(self, rhs: ArrayLike, sparsity: str = "bta") -> ArrayLike:
        """Solve linear system using Cholesky factor."""

        if sparsity == "bta":
            return pobtas(
                self.L_diagonal_blocks,
                self.L_lower_diagonal_blocks,
                self.L_arrow_bottom_blocks,
                self.L_arrow_tip_block,
                rhs,
            )
        else:
            raise NotImplementedError("Solve is only supported for BTA sparsity")

    def logdet(self, sparsity: str = "bta") -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        logdet: float = 0.0

        for i in range(self.n_diagonal_blocks):
            logdet += np.sum(np.log(self.L_diagonal_blocks[i].diagonal()))

        if sparsity == "bta":
            logdet += np.sum(np.log(self.L_arrow_tip_block.diagonal()))

        return 2 * logdet

    def _sparray_to_structured(self, A: sparray, sparsity: str) -> None:
        """Map sparray to BT or BTA."""

        A_csr = A.tocsr()

        for i in range(self.n_diagonal_blocks):

            csr_slice = A_csr[
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ]

            self.A_diagonal_blocks[i, :, :] = csr_slice.todense()

            if i < self.n_diagonal_blocks - 1:
                csr_slice = A_csr[
                    (i + 1)
                    * self.diagonal_blocksize : (i + 2)
                    * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ]

                self.A_lower_diagonal_blocks[i, :, :] = csr_slice.todense()

            if sparsity == "bta":
                csr_slice = A_csr[
                    -self.arrowhead_blocksize :,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ]

                self.A_arrow_bottom_blocks[i, :, :] = csr_slice.todense()

        if sparsity == "bta":
            csr_slice = A_csr[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :]

            self.A_arrow_tip_block[:, :] = csr_slice.todense()

    def full_inverse(self) -> ArrayLike:
        """Compute full inverse of A."""
        raise NotImplementedError

    def get_selected_inverse(self) -> sparray:
        """extract values of the inverse of A that are nonzero in A."""
        raise NotImplementedError

    def selected_inverse(self) -> sparray:
        """Compute inverse of nonzero sparsity pattern of L."""

        raise NotImplementedError

    def get_L(self, sparsity: str = "bta") -> ArrayLike:
        """Get L as a dense array."""

        n = self.diagonal_blocksize * self.n_diagonal_blocks

        if sparsity == "bta":
            n += self.arrowhead_blocksize

        L = np.zeros(
            (n, n),
            dtype=self.A_diagonal_blocks.dtype,
        )

        for i in range(self.n_diagonal_blocks):

            L[
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ] = self.A_diagonal_blocks[i, :, :]

            if i < self.n_diagonal_blocks - 1:
                L[
                    (i + 1)
                    * self.diagonal_blocksize : (i + 2)
                    * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ] = self.A_lower_diagonal_blocks[i, :, :]

            if sparsity == "bta":
                L[
                    -self.arrowhead_blocksize :,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ] = self.A_arrow_bottom_blocks[i, :, :]

        if sparsity == "bta":
            L[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :] = (
                self.A_arrow_tip_block[:, :]
            )

        return L
