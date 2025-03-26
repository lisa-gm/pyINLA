# Copyright 2024-2025 pyINLA authors. All rights reserved.

from warnings import warn

from pyinla import NDArray, sp, xp
from pyinla.configs.pyinla_config import SolverConfig
from pyinla.core.solver import Solver
from pyinla.utils import print_msg

try:
    from serinv.algs import pobtaf, pobtas, pobtasi, pobtf, pobts, pobtsi
except ImportError as e:
    warn(f"The serinv package is required to use the SerinvSolver: {e}")


import time


class SerinvSolver(Solver):
    """Serinv Solver class."""

    def __init__(
        self,
        config: SolverConfig,
        diagonal_blocksize: int,
        n_diag_blocks: int,
        arrowhead_blocksize: int = 0,
        **kwargs,
    ) -> None:
        """Initializes the SerinV solver."""
        super().__init__(config)

        self.diagonal_blocksize: int = diagonal_blocksize
        self.arrowhead_blocksize: int = arrowhead_blocksize
        self.n_diag_blocks: int = n_diag_blocks

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

    def cholesky(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
    ) -> None:
        """Compute Cholesky factor of input matrix."""
        self._spmatrix_to_structured(A, sparsity)

        if sparsity == "bta":
            pobtaf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
            )
        elif sparsity == "bt":
            pobtf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
            )
        else:
            raise ValueError(
                f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
            )

    def solve(
        self,
        rhs: NDArray,
        sparsity: str,
    ) -> NDArray:
        """Solve linear system using Cholesky factor."""
        if sparsity == "bta":
            pobtas(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
                rhs,
                trans="N",
            )
            pobtas(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
                rhs,
                trans="C",
            )
        elif sparsity == "bt":
            pobts(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                rhs,
                trans="N",
            )
            pobts(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                rhs,
                trans="C",
            )
        else:
            raise ValueError(
                f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
            )

        return rhs

    def logdet(
        self,
        sparsity: str,
    ) -> float:
        """Compute logdet of input matrix using Cholesky factor."""
        logdet: float = 0.0
        for i in range(self.n_diag_blocks):
            logdet += xp.sum(xp.log(self.A_diagonal_blocks[i].diagonal()))

        if sparsity == "bta":
            logdet += xp.sum(xp.log(self.A_arrow_tip_block.diagonal()))

        # print(f"sequential! logdet: {logdet}")
        # exit()

        return 2 * logdet

    def selected_inversion(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
    ) -> None:
        """Compute selected inversion of input matrix using Cholesky factor."""
        # self._spmatrix_to_structured(A, sparsity)

        if sparsity == "bta":
            pobtasi(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
            )
        elif sparsity == "bt":
            pobtsi(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
            )
        else:
            raise ValueError(
                f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
            )

    def _spmatrix_to_structured(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
    ) -> None:
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

            if sparsity == "bta":
                csc_slice = A_csc[
                    -self.arrowhead_blocksize :,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ]
                self.A_arrow_bottom_blocks[i, :, :] = csc_slice.todense()

        if sparsity == "bta":
            csc_slice = A_csc[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :]
            self.A_arrow_tip_block[:, :] = csc_slice.todense()

    def _structured_to_spmatrix(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
    ) -> sp.sparse.spmatrix:
        """Map BT or BTA matrix to sp.spmatrix using sparsity pattern provided in A."""

        # A is assumed to be symmetric, only use lower triangular part
        B = sp.sparse.csc_matrix(sp.sparse.tril(sp.sparse.csc_matrix(A)))

        for i in range(self.n_diag_blocks):
            # Extract the sparsity pattern of the current Diagonal, Lower, and Arrowhead blocks
            row_diag, col_diag = B[
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ].nonzero()
            B[
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ] = sp.sparse.coo_matrix(
                (self.A_diagonal_blocks[i, row_diag, col_diag], (row_diag, col_diag)),
                shape=B[
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].shape,
            )

            if i < self.n_diag_blocks - 1:
                row_lower, col_lower = B[
                    (i + 1)
                    * self.diagonal_blocksize : (i + 2)
                    * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].nonzero()
                B[
                    (i + 1)
                    * self.diagonal_blocksize : (i + 2)
                    * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ] = sp.sparse.coo_matrix(
                    (
                        self.A_lower_diagonal_blocks[i, row_lower, col_lower],
                        (row_lower, col_lower),
                    ),
                    shape=B[
                        (i + 1)
                        * self.diagonal_blocksize : (i + 2)
                        * self.diagonal_blocksize,
                        i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                    ].shape,
                )

            if sparsity == "bta":
                row_arrow, col_arrow = B[
                    -self.arrowhead_blocksize :,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].nonzero()
                B[
                    -self.arrowhead_blocksize :,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ] = sp.sparse.coo_matrix(
                    (
                        self.A_arrow_bottom_blocks[i, row_arrow, col_arrow],
                        (row_arrow, col_arrow),
                    ),
                    shape=B[
                        -self.arrowhead_blocksize :,
                        i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                    ].shape,
                )

        if sparsity == "bta":
            row_arrow_tip, col_arrow_tip = B[
                -self.arrowhead_blocksize :, -self.arrowhead_blocksize :
            ].nonzero()
            B[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :] = (
                sp.sparse.coo_matrix(
                    (
                        self.A_arrow_tip_block[row_arrow_tip, col_arrow_tip],
                        (row_arrow_tip, col_arrow_tip),
                    ),
                    shape=B[
                        -self.arrowhead_blocksize :, -self.arrowhead_blocksize :
                    ].shape,
                )
            )

        # Symmetrize B
        B = B + sp.sparse.tril(B, k=-1).T

        return B
