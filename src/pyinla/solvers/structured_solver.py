# Copyright 2024-2025 pyINLA authors. All rights reserved.

from warnings import warn

from pyinla import NDArray, sp, xp
from pyinla.configs.pyinla_config import SolverConfig
from pyinla.core.solver import Solver
from pyinla.utils import print_msg

try:
    from serinv.algs import pobtaf, pobtas, pobtf, pobts
except ImportError as e:
    warn(f"The serinv package is required to use the SerinvSolver: {e}")


class SerinvSolver(Solver):
    """Serinv Solver class."""

    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the SerinV solver."""
        super().__init__(config)

        self.diagonal_blocksize: int = kwargs.get("diagonal_blocksize", None)
        if self.diagonal_blocksize is None:
            raise KeyError("Missing required keyword argument: 'diagonal_blocksize'")

        self.arrowhead_blocksize: int = kwargs.get("arrowhead_blocksize", 0)

        self.n_diag_blocks: int = kwargs.get("n_diag_blocks", None)
        if self.n_diag_blocks is None:
            raise KeyError("Missing required keyword argument: 'n_diag_blocks'")

        # --- Initialize memory for BTA-array storage
        self.A_diagonal_blocks: NDArray = xp.empty(
            (self.n_diag_blocks, self.diagonal_blocksize, self.diagonal_blocksize),
            dtype=xp.float64,
        )
        self.A_lower_diagonal_blocks: NDArray = xp.empty(
            (
                self.n_diag_blocks - 1,
                self.diagonal_blocksize,
                self.diagonal_blocksize,
            ),
            dtype=xp.float64,
        )

        self.A_arrow_bottom_blocks: NDArray = None
        self.A_arrow_tip_block: NDArray = None

        if self.arrowhead_blocksize > 0:
            self.A_arrow_bottom_blocks = xp.empty(
                (self.n_diag_blocks, self.arrowhead_blocksize, self.diagonal_blocksize),
                dtype=xp.float64,
            )
            self.A_arrow_tip_block = xp.empty(
                (self.arrowhead_blocksize, self.arrowhead_blocksize),
                dtype=xp.float64,
            )

        # Print the allocated memory for the BTA-array
        total_bytes: int = (
            self.A_diagonal_blocks.nbytes
            + self.A_lower_diagonal_blocks.nbytes
            + self.A_arrow_bottom_blocks.nbytes
            + self.A_arrow_tip_block.nbytes
        )
        total_gb: int = total_bytes / (1024**3)
        print_msg(f"Allocated memory for SerinvSolver: {total_gb:.2f} GB", flush=True)

    def cholesky(self, A: sp.sparse.spmatrix) -> None:
        """Compute Cholesky factor of input matrix."""

        self._spmatrix_to_structured(A)

        if self.A_arrow_bottom_blocks is not None:
            pobtaf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
            )
        else:
            pobtf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
            )

    def solve(self, rhs: NDArray) -> NDArray:
        """Solve linear system using Cholesky factor."""

        if self.A_arrow_bottom_blocks is not None:
            pobtas(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
                rhs,
            )
        else:
            pobts(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                rhs,
            )

        return rhs

    def logdet(self) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        logdet: float = 0.0

        for i in range(self.n_diag_blocks):
            logdet += xp.sum(xp.log(self.A_diagonal_blocks[i].diagonal()))

        logdet += xp.sum(xp.log(self.A_arrow_tip_block.diagonal()))

        return 2 * logdet

    def selected_inversion(self, **kwargs) -> NDArray:
        """Compute selected inversion of input matrix using Cholesky factor."""

    def _spmatrix_to_structured(self, A: sp.sparse.spmatrix) -> None:
        """Map sp.spmatrix to BT or BTA."""

        A_csc = sp.sparse.csc_matrix(A)

        for i in range(self.n_diag_blocks):
            csc_slice = A_csc[
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ]

            self.A_diagonal_blocks[i, :, :] = csc_slice.todense()

            if i < self.n_diag_blocks - 1:
                csc_slice = A_csc[
                    (i + 1)
                    * self.diagonal_blocksize : (i + 2)
                    * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ]

                self.A_lower_diagonal_blocks[i, :, :] = csc_slice.todense()

            if self.arrowhead_blocksize is not None:
                csc_slice = A_csc[
                    -self.arrowhead_blocksize :,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ]

                self.A_arrow_bottom_blocks[i, :, :] = csc_slice.todense()

        if self.arrowhead_blocksize is not None:
            csc_slice = A_csc[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :]
            self.A_arrow_tip_block[:, :] = csc_slice.todense()

    def _structured_to_spmatrix(
        self, sparsity_pattern: sp.sparse.spmatrix
    ) -> sp.sparse.spmatrix:
        """Map BT or BTA matrix to sp.spmatrix using pattern provided in sparsity_pattern."""

        sparsity_pattern_csc = sp.sparse.csc_matrix(sparsity_pattern)
        # get lower triangular part
        sparsity_pattern_csc = sp.sparse.tril(sparsity_pattern_csc)

        for col in range(sparsity_pattern.shape[1]):
            start = sparsity_pattern_csc.indptr[col]
            end = sparsity_pattern_csc.indptr[col + 1]
            for i in range(start, end):
                row = sparsity_pattern_csc.indices[i]

                block_i = row // self.diagonal_blocksize
                block_j = col // self.diagonal_blocksize
                local_i = row % self.diagonal_blocksize
                local_j = col % self.diagonal_blocksize

                if block_i == block_j and block_i < self.n_diag_blocks:
                    self.A_diagonal_blocks[block_i, local_i, local_j] = (
                        sparsity_pattern_csc[row, col]
                    )
                elif block_i + 1 == block_j and block_i < self.n_diag_blocks - 1:
                    print("block_i: ", block_i, "block_j: ", block_j)
                    self.A_lower_diagonal_blocks[block_i, local_i, local_j] = (
                        sparsity_pattern_csc[row, col]
                    )
                # arrowhead
                elif block_i == self.n_diag_blocks and block_j < self.n_diag_blocks:
                    self.A_arrow_bottom_blocks[block_j, local_i, local_j] = (
                        sparsity_pattern_csc[row, col]
                    )
                elif block_i == self.n_diag_blocks and block_j == self.n_diag_blocks:
                    self.A_arrow_tip_block[local_i, local_j] = sparsity_pattern_csc[
                        row, col
                    ]

        print("output matrix: ", sparsity_pattern_csc.todense())
        return sparsity_pattern_csc


import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(41)

path = os.path.dirname(__file__)


def diagonally_dominant_bta(
    n: int,
    b: int,
    a: int,
    direction: str = "downward",
    device_arr: bool = False,
    symmetric: bool = True,
    seed: int = 63,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:

    xp.random.seed(seed)

    A_diagonal_blocks = xp.random.rand(n, b, b)
    A_lower_diagonal_blocks = xp.random.rand(n - 1, b, b)
    A_upper_diagonal_blocks = xp.random.rand(n - 1, b, b)
    A_arrow_tip_block = xp.random.rand(a, a)

    if direction == "downward" or direction == "down-middleward":
        A_lower_arrow_blocks = xp.random.rand(n, a, b)
        A_upper_arrow_blocks = xp.random.rand(n, b, a)
    elif direction == "upward" or direction == "up-middleward":
        A_lower_arrow_blocks = xp.random.rand(n, b, a)
        A_upper_arrow_blocks = xp.random.rand(n, a, b)

    if symmetric:
        for n_i in range(n):
            A_diagonal_blocks[n_i] = A_diagonal_blocks[n_i] + A_diagonal_blocks[n_i].T

            if n_i < n - 1:
                A_upper_diagonal_blocks[n_i] = A_lower_diagonal_blocks[n_i].T

            A_upper_arrow_blocks[n_i] = A_lower_arrow_blocks[n_i].T

        A_arrow_tip_block[:] = A_arrow_tip_block[:] + A_arrow_tip_block[:].T

    A_arrow_tip_block[:] += xp.diag(xp.sum(xp.abs(A_arrow_tip_block[:]), axis=1))

    if direction == "downward" or direction == "down-middleward":
        for n_i in range(n):
            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_diagonal_blocks[n_i]), axis=1)
            )

            if n_i > 0:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_lower_diagonal_blocks[n_i - 1]), axis=1)
                )

            if n_i < n - 1:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_upper_diagonal_blocks[n_i]), axis=1)
                )

            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_upper_arrow_blocks[n_i]), axis=1)
            )

            A_arrow_tip_block[:] += xp.diag(
                xp.sum(xp.abs(A_lower_arrow_blocks[n_i]), axis=1)
            )

    elif direction == "upward" or direction == "up-middleward":
        for n_i in range(n - 1, -1, -1):
            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_diagonal_blocks[n_i]), axis=1)
            )

            if n_i < n - 1:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_upper_diagonal_blocks[n_i]), axis=1)
                )

            if n_i > 0:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_lower_diagonal_blocks[n_i - 1]), axis=1)
                )

            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_lower_arrow_blocks[n_i]), axis=1)
            )

            A_arrow_tip_block[:] += xp.diag(
                xp.sum(xp.abs(A_upper_arrow_blocks[n_i]), axis=1)
            )

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_diagonal_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    )


def bta_to_dense(
    A_diagonal_blocks: NDArray,
    A_lower_diagonal_blocks: NDArray,
    A_lower_arrow_blocks: NDArray,
    A_upper_diagonal_blocks: NDArray,
    A_upper_arrow_blocks: NDArray,
    A_arrow_tip_block: NDArray,
    direction: str = "downward",
) -> NDArray:

    n = A_diagonal_blocks.shape[0]
    b = A_diagonal_blocks.shape[1]
    a = A_arrow_tip_block.shape[0]

    A = xp.zeros((n * b + a, n * b + a), dtype=A_diagonal_blocks.dtype)

    if direction == "downward" or direction == "down-middleward":
        for n_i in range(n):
            A[n_i * b : (n_i + 1) * b, n_i * b : (n_i + 1) * b] = A_diagonal_blocks[n_i]
            if n_i > 0:
                A[n_i * b : (n_i + 1) * b, (n_i - 1) * b : n_i * b] = (
                    A_lower_diagonal_blocks[n_i - 1]
                )
            if n_i < n - 1:
                A[n_i * b : (n_i + 1) * b, (n_i + 1) * b : (n_i + 2) * b] = (
                    A_upper_diagonal_blocks[n_i]
                )
            A[n_i * b : (n_i + 1) * b, -a:] = A_upper_arrow_blocks[n_i]
            A[-a:, n_i * b : (n_i + 1) * b] = A_lower_arrow_blocks[n_i]
        A[-a:, -a:] = A_arrow_tip_block[:]

    if direction == "upward" or direction == "up-middleward":
        for n_i in range(n - 1, -1, -1):
            A[n_i * b + a : (n_i + 1) * b + a, n_i * b + a : (n_i + 1) * b + a] = (
                A_diagonal_blocks[n_i]
            )
            if n_i > 0:
                A[n_i * b + a : (n_i + 1) * b + a, (n_i - 1) * b + a : n_i * b + a] = (
                    A_lower_diagonal_blocks[n_i - 1]
                )
            if n_i < n - 1:
                A[
                    n_i * b + a : (n_i + 1) * b + a,
                    (n_i + 1) * b + a : (n_i + 2) * b + a,
                ] = A_upper_diagonal_blocks[n_i]
            A[n_i * b + a : (n_i + 1) * b + a, :a] = A_lower_arrow_blocks[n_i]
            A[:a, n_i * b + a : (n_i + 1) * b + a] = A_upper_arrow_blocks[n_i]
        A[:a, :a] = A_arrow_tip_block[:]

    return A


if __name__ == "__main__":

    diagonal_blocksize = 2
    arrowhead_blocksize = 1
    n_diag_blocks = 5

    # Define the required keyword arguments
    kwargs = {
        "diagonal_blocksize": diagonal_blocksize,
        "arrowhead_blocksize": arrowhead_blocksize,
        "n_diag_blocks": n_diag_blocks,
    }

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_diagonal_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    ) = diagonally_dominant_bta(n_diag_blocks, diagonal_blocksize, arrowhead_blocksize)

    A_dense = bta_to_dense(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_diagonal_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    )

    print("A_dense: \n", A_dense)

    # save spy(A_dense)
    plt.figure()
    plt.spy(A_dense)
    plt.savefig("A_dense.png")

    # save as coo
    A_coo = sp.sparse.coo_matrix(A_dense)

    nnz_A = A_coo.nnz

    # # generate random sparse matrix
    # n = 5

    # A = sp.sparse.random(n, n, density=0.2)
    # A = A @ A.T
    # # add diagonal
    # A = A + n * sp.sparse.eye(n, n)

    # print("A: \n", A.todense())

    # Create a SolverConfig instance
    config = SolverConfig()

    # Create an instance of the SerinvSolver class
    solver = SerinvSolver(config, **kwargs)

    # Now you can use the solver instance
    print("Solver initialized successfully.")

    solver.cholesky(A_coo)

    A_inv = solver._structured_to_spmatrix(A_coo)

    plt.figure()
    plt.spy(A_inv)
    plt.savefig("A_inv.png")
