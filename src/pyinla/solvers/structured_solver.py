# Copyright 2024 pyINLA authors. All rights reserved.

import time

import numpy as np
from cupyx.profiler import time_range
from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
from scipy.linalg import cholesky as scipy_chol
from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.sparse import sparray

from pyinla import ArrayLike
from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.core.solver import Solver
from pyinla.utils.other_utils import print_mpi

try:
    from serinv import pobtaf, pobtas, pobtf

    # from serinv.cupyfix.cholesky_lowerfill import cholesky_lowerfill
except ImportError:
    print_mpi("Serinv not installed. Please install serinv to use SerinvSolver.")


try:
    import cupy as cp
    import cupyx as cpx
except ImportError:
    print_mpi("CuPy not installed. Please install cupy to use GPU compute.")


class SerinvSolver(Solver):
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

        # --- Initialize memory for BTA-array storage
        self.A_diagonal_blocks = np.empty(
            (self.n_diagonal_blocks, self.diagonal_blocksize, self.diagonal_blocksize),
            dtype=self.dtype,
        )

        if self.n_diagonal_blocks > 0:
            self.A_lower_diagonal_blocks = np.empty(
                (
                    self.n_diagonal_blocks - 1,
                    self.diagonal_blocksize,
                    self.diagonal_blocksize,
                ),
                dtype=self.dtype,
            )

        else:
            self.A_lower_diagonal_blocks = np.empty(
                (
                    0,
                    0,
                    0,
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
        print_mpi(f"Allocated memory for BTA-array: {total_gb:.2f} GB", flush=True)

        # --- Make aliases for L
        self.L_diagonal_blocks = self.A_diagonal_blocks
        self.L_lower_diagonal_blocks = self.A_lower_diagonal_blocks
        self.L_arrow_bottom_blocks = self.A_arrow_bottom_blocks
        self.L_arrow_tip_block = self.A_arrow_tip_block

    @time_range()
    def cholesky(self, A: sparray, sparsity: str = "bta") -> None:
        """Compute Cholesky factor of input matrix."""

        if sparsity == "bta":
            self._sparray_to_structured(A, sparsity="bta")

            # with time_range('callPobtafBTA', color_id=0):
            tic = time.perf_counter()
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
            toc = time.perf_counter()
            print_mpi(
                "                 pobtaf Q_conditional time:", toc - tic, flush=True
            )

        elif sparsity == "bt":
            self._sparray_to_structured(A, sparsity="bt")

            with time_range("callPobtafBT", color_id=0):
                (
                    self.L_diagonal_blocks,
                    self.L_lower_diagonal_blocks,
                ) = pobtf(
                    self.A_diagonal_blocks,
                    self.A_lower_diagonal_blocks,
                )

            # Factorize the unconnected tip of the arrow
            # with time_range('npCholesky', color_id=0):
            self.L_arrow_tip_block[:, :] = np.linalg.cholesky(self.A_arrow_tip_block)

        elif sparsity == "d":
            self._sparray_to_structured(A, sparsity="d")
            self.L_arrow_tip_block[:, :] = scipy_chol(
                self.A_arrow_tip_block, lower=True
            )

        else:
            print("Sparsity pattern not supported: ", sparsity)
            raise ValueError

    @time_range()
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

        elif sparsity == "d":
            y = scipy_solve_triangular(self.L_arrow_tip_block, rhs, lower=True)
            x = scipy_solve_triangular(self.L_arrow_tip_block.T, y, lower=False)
            return x

        else:
            raise NotImplementedError(
                "Solve is only supported for BTA sparsity and dense matrix"
            )

    @time_range()
    def logdet(self) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        logdet: float = 0.0

        for i in range(self.n_diagonal_blocks):
            logdet += np.sum(np.log(self.L_diagonal_blocks[i].diagonal()))

        logdet += np.sum(np.log(self.L_arrow_tip_block.diagonal()))

        return 2 * logdet

    @time_range()
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

        # Copy the arrow tip block
        csr_slice = A_csr[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :]
        self.A_arrow_tip_block[:, :] = csr_slice.todense()

    def get_L(self, sparsity: str = "bta") -> ArrayLike:
        """Get L as a dense array."""

        n = self.diagonal_blocksize * self.n_diagonal_blocks + self.arrowhead_blocksize

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

        L[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :] = (
            self.A_arrow_tip_block[:, :]
        )

        return L
