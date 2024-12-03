# Copyright 2024 pyINLA authors. All rights reserved.


from pyinla import xp, sp, NDArray

from pyinla.core.solver import Solver
from pyinla.core.pyinla_config import SolverConfig
from pyinla.utils import print_msg

try:
    from serinv.algs import pobtaf, pobtas, pobtf, pobts
except:
    raise ImportError("The serinv package is required to use the SerinvSolver.")


class SerinvSolver(Solver):
    """Serinv Solver class."""

    def __init__(
        self,
        solver_config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the SerinV solver."""
        super().__init__(solver_config)

        self.diagonal_blocksize: int = kwargs.get("diagonal_blocksize")
        if self.diagonal_blocksize is None:
            raise KeyError("Missing required keyword argument: 'diagonal_blocksize'")

        self.arrowhead_size: int = kwargs.get("arrowhead_size", 0)

        self.n_diag_blocks: int = kwargs.get("n_diag_blocks")
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

        if self.arrowhead_size > 0:
            self.A_arrow_bottom_blocks = xp.empty(
                (self.n_diag_blocks, self.arrowhead_size, self.diagonal_blocksize),
                dtype=xp.float64,
            )
            self.A_arrow_tip_block = xp.empty(
                (self.arrowhead_size, self.arrowhead_size),
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
            return pobtas(
                self.L_diagonal_blocks,
                self.L_lower_diagonal_blocks,
                self.L_arrow_bottom_blocks,
                self.L_arrow_tip_block,
                rhs,
            )
        else:
            pobts(
                self.L_diagonal_blocks,
                self.L_lower_diagonal_blocks,
                rhs,
            )

    def logdet(self) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        logdet: float = 0.0

        for i in range(self.n_diag_blocks):
            logdet += xp.sum(xp.log(self.L_diagonal_blocks[i].diagonal()))

        logdet += xp.sum(xp.log(self.L_arrow_tip_block.diagonal()))

        return 2 * logdet

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

            if self.arrowhead_size is not None:
                csc_slice = A_csc[
                    -self.arrowhead_size :,
                    i * self.diagonal_blocksize : (i + 1) * self.diagonal_blocksize,
                ]

                self.A_arrow_bottom_blocks[i, :, :] = csc_slice.todense()

        if self.arrowhead_size is not None:
            csc_slice = A_csc[-self.arrowhead_size :, -self.arrowhead_size :]
            self.A_arrow_tip_block[:, :] = csc_slice.todense()
