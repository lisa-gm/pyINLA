# Copyright 2024-2025 pyINLA authors. All rights reserved.

from warnings import warn

from pyinla import NDArray, sp, xp, backend_flags
from pyinla.configs.pyinla_config import SolverConfig
from pyinla.core.solver import Solver
from pyinla.utils import print_msg

try:
    from serinv.algs import pobtaf, pobtas, pobtasi, pobtf, pobts, pobtsi
except ImportError as e:
    warn(f"The serinv package is required to use the SerinvSolver: {e}")

if backend_flags["cupy_avail"]:
    from cupyx.profiler import time_range

import time


class SerinvSolver(Solver):
    """Serinv Solver class."""

    @time_range()
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

    @time_range()
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

    @time_range()
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

    @time_range()
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

        return 2 * logdet

    @time_range()
    def selected_inversion(
        self,
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

    @time_range()
    def _spmatrix_to_structured(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
    ) -> None:
        """Map sp.spmatrix to BT or BTA."""
        # About 3x faster...
        A_csc = sp.sparse.csc_matrix(A)

        self.A_diagonal_blocks[:] = 0.0
        self.A_lower_diagonal_blocks[:] = 0.0
        self.A_arrow_bottom_blocks[:] = 0.0
        self.A_arrow_tip_block[:] = 0.0

        for i in range(self.n_diag_blocks):
            block_slice = A_csc[
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ].tocoo()
            self.A_diagonal_blocks[i][
                block_slice.row, block_slice.col
            ] = block_slice.data

            if i < self.n_diag_blocks - 1:
                block_slice = A_csc[
                    (i + 1)
                    * self.diagonal_blocksize : (i + 2)
                    * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].tocoo()
                self.A_lower_diagonal_blocks[i][
                    block_slice.row, block_slice.col
                ] = block_slice.data

            block_slice = A_csc[
                -self.arrowhead_blocksize :,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ].tocoo()
            self.A_arrow_bottom_blocks[i][
                block_slice.row, block_slice.col
            ] = block_slice.data

        block_slice = A_csc[
            -self.arrowhead_blocksize :, -self.arrowhead_blocksize :
        ].tocoo()
        self.A_arrow_tip_block[block_slice.row, block_slice.col] = block_slice.data

    @time_range()
    def _structured_to_spmatrix(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
    ) -> sp.sparse.spmatrix:
        """Map BT or BTA matrix to sp.spmatrix using sparsity pattern provided in A."""

        # A is assumed to be symmetric, only use lower triangular part
        B = sp.sparse.csc_matrix(sp.sparse.tril(sp.sparse.csc_matrix(A)))

        data = []
        rows = []
        cols = []

        for i in range(self.n_diag_blocks):
            # Extract the sparsity pattern of the current Diagonal, Lower, and Arrowhead blocks
            B_coo = B[
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
            ].tocoo()
            row_diag, col_diag = B_coo.row, B_coo.col
            data.append(self.A_diagonal_blocks[i, row_diag, col_diag].flatten())
            rows.append(i * self.diagonal_blocksize + row_diag)
            cols.append(i * self.diagonal_blocksize + col_diag)

            if i < self.n_diag_blocks - 1:
                B_coo = B[
                    (i + 1)
                    * self.diagonal_blocksize : (i + 2)
                    * self.diagonal_blocksize,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].tocoo()
                row_lower, col_lower = B_coo.row, B_coo.col
                data.append(
                    self.A_lower_diagonal_blocks[i, row_lower, col_lower].flatten()
                )
                rows.append((i + 1) * self.diagonal_blocksize + row_lower)
                cols.append(i * self.diagonal_blocksize + col_lower)

            if sparsity == "bta":
                B_coo = B[
                    -self.arrowhead_blocksize :,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ].tocoo()
                row_arrow, col_arrow = B_coo.row, B_coo.col
                data.append(
                    self.A_arrow_bottom_blocks[i, row_arrow, col_arrow].flatten()
                )
                rows.append(self.n_diag_blocks * self.diagonal_blocksize + row_arrow)
                cols.append(i * self.diagonal_blocksize + col_arrow)

        if sparsity == "bta":
            B_coo = B[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :].tocoo()
            row_arrow_tip, col_arrow_tip = B_coo.row, B_coo.col
            data.append(self.A_arrow_tip_block[row_arrow_tip, col_arrow_tip].flatten())
            rows.append(self.n_diag_blocks * self.diagonal_blocksize + row_arrow_tip)
            cols.append(self.n_diag_blocks * self.diagonal_blocksize + col_arrow_tip)

        # Create the sparse matrix from the data, rows, and cols
        data = xp.concatenate(data)
        rows = xp.concatenate(rows)
        cols = xp.concatenate(cols)

        B_out = sp.sparse.coo_matrix((data, (rows, cols)), shape=B.shape).tocsc()

        # Symmetrize B
        B_out = B_out + sp.sparse.tril(B_out, k=-1).T

        return B_out
