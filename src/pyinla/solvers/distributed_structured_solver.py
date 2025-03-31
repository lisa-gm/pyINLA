# Copyright 2024-2025 pyINLA authors. All rights reserved.

from warnings import warn

import numpy as np

from pyinla import NDArray, sp, xp, backend_flags
from pyinla.configs.pyinla_config import SolverConfig
from pyinla.core.solver import Solver
from pyinla.utils import print_msg, allreduce, allgatherv

if backend_flags["mpi_avail"]:
    from mpi4py.MPI import Comm as mpi_comm
    from mpi4py import MPI
else:
    mpi_comm = None

try:
    from serinv.utils import allocate_pobtax_permutation_buffers
    from serinv.wrappers import ppobtaf, ppobtasi, ppobtas, allocate_pobtars
    from serinv.wrappers import ppobtf, ppobtsi, ppobts
except ImportError as e:
    warn(f"The serinv package is required to use the SerinvSolver: {e}")


class DistSerinvSolver(Solver):
    """Serinv Solver class."""

    def __init__(
        self,
        config: SolverConfig,
        diagonal_blocksize: int,
        n_diag_blocks: int,
        comm: mpi_comm,
        arrowhead_blocksize: int = 0,
        **kwargs,
    ) -> None:
        """Initializes the SerinV solver."""
        super().__init__(config)

        self.diagonal_blocksize: int = diagonal_blocksize
        self.arrowhead_blocksize: int = arrowhead_blocksize
        self.n_diag_blocks: int = n_diag_blocks
        self.comm: mpi_comm = comm
        self.rank: int = self.comm.Get_rank()
        self.comm_size: int = self.comm.Get_size()

        # Allocating the local slices of the system matrix
        self.n_locals = [n_diag_blocks // self.comm_size] * self.comm_size
        remainder = n_diag_blocks % self.comm_size
        self.n_locals[0] += remainder

        print_msg(
            f"Distributed solver slicing: {self.n_locals}",
            flush=True,
        )

        # --- Initialize memory for BTA-array storage
        self.A_diagonal_blocks: NDArray = xp.empty(
            (
                self.n_locals[self.rank],
                self.diagonal_blocksize,
                self.diagonal_blocksize,
            ),
            dtype=xp.float64,
        )
        # Here is the "arrowhead" slicing
        if self.rank == self.comm_size - 1:
            self.A_lower_diagonal_blocks: NDArray = xp.empty(
                (
                    self.n_locals[self.rank] - 1,
                    self.diagonal_blocksize,
                    self.diagonal_blocksize,
                ),
                dtype=xp.float64,
            )
        else:
            self.A_lower_diagonal_blocks: NDArray = xp.empty(
                (
                    self.n_locals[self.rank],
                    self.diagonal_blocksize,
                    self.diagonal_blocksize,
                ),
                dtype=xp.float64,
            )
        self.A_arrow_bottom_blocks: NDArray = None
        self.A_arrow_tip_block: NDArray = None
        if self.arrowhead_blocksize > 0:
            self.A_arrow_bottom_blocks = xp.empty(
                (
                    self.n_locals[self.rank],
                    self.arrowhead_blocksize,
                    self.diagonal_blocksize,
                ),
                dtype=xp.float64,
            )
            self.A_arrow_tip_block = xp.empty(
                (self.arrowhead_blocksize, self.arrowhead_blocksize),
                dtype=xp.float64,
            )
        n_rhs = 1
        self.B: NDArray = xp.empty(
            (
                self.n_locals[self.rank] * self.diagonal_blocksize
                + self.arrowhead_blocksize,
                n_rhs,
            ),
            dtype=xp.float64,
        )

        # --- Allocate reduced system and distribution buffers
        self.buffer = allocate_pobtax_permutation_buffers(
            self.A_diagonal_blocks,
        )
        self.pobtars: dict = allocate_pobtars(
            A_diagonal_blocks=self.A_diagonal_blocks,
            A_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
            A_lower_arrow_blocks=self.A_arrow_bottom_blocks,
            A_arrow_tip_block=self.A_arrow_tip_block,
            B=self.B,
            comm=self.comm,
            array_module=xp.__name__,
            strategy="allgather",
        )

        # Print the allocated memory for the BTA-array
        bytes_pobtars: int = (
            self.buffer.nbytes
            + self.pobtars["A_diagonal_blocks"].nbytes
            + self.pobtars["A_lower_diagonal_blocks"].nbytes
            + self.pobtars["A_lower_arrow_blocks"].nbytes
            + self.pobtars["A_arrow_tip_block"].nbytes
            + self.pobtars["B"].nbytes
        )
        bytes_local_system: int = (
            self.A_diagonal_blocks.nbytes
            + self.A_lower_diagonal_blocks.nbytes
            + self.A_arrow_bottom_blocks.nbytes
            + self.A_arrow_tip_block.nbytes
        )
        total_bytes: int = bytes_pobtars + bytes_local_system
        total_gb: int = total_bytes / (1024**3)
        print_msg(
            f"Local allocated memory for DistSerinvSolver: {total_gb:.2f} GB",
            flush=True,
        )

    def cholesky(
        self,
        A: sp.sparse.spmatrix,
        sparsity: str,
    ) -> None:
        """Compute Cholesky factor of input matrix."""
        # Reset the tip block for reccurrent calls
        self._spmatrix_to_structured(A, sparsity)

        if sparsity == "bta":
            ppobtaf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                self.A_arrow_bottom_blocks,
                self.A_arrow_tip_block,
                buffer=self.buffer,
                pobtars=self.pobtars,
                comm=self.comm,
                strategy="allgather",
            )
        elif sparsity == "bt":
            ppobtf(
                self.A_diagonal_blocks,
                self.A_lower_diagonal_blocks,
                buffer=self.buffer,
                pobtrs=self.pobtars,
                comm=self.comm,
                strategy="allgather",
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
        self._slice_rhs(rhs, sparsity)

        if sparsity == "bta":
            ppobtas(
                L_diagonal_blocks=self.A_diagonal_blocks,
                L_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                L_lower_arrow_blocks=self.A_arrow_bottom_blocks,
                L_arrow_tip_block=self.A_arrow_tip_block,
                B=self.B,
                buffer=self.buffer,
                pobtars=self.pobtars,
                comm=self.comm,
                strategy="allgather",
            )
        elif sparsity == "bt":
            ppobts(
                L_diagonal_blocks=self.A_diagonal_blocks,
                L_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                B=self.B[-self.arrowhead_blocksize :],
                buffer=self.buffer,
                pobtars=self.pobtars,
                comm=self.comm,
                strategy="allgather",
            )
        else:
            raise ValueError(
                f"Unknown sparsity pattern: {sparsity}. Use 'bt' or 'bta'."
            )

        self._gather_rhs(rhs, sparsity)
        return rhs

    def logdet(
        self,
        sparsity: str,
    ) -> float:
        """Compute logdet of input matrix using Cholesky factor."""
        logdet = xp.array(0.0, dtype=xp.float64)

        if self.rank == 0:
            # Do its local blocks
            for i in range(self.n_locals[self.rank] - 1):
                logdet += xp.sum(xp.log(self.A_diagonal_blocks[i].diagonal()))

            # Rank 0 do the reduced system; The loop start from 1 because of the
            # AllGather strategy and the size of the reduced system associated.
            _n = self.pobtars["A_diagonal_blocks"].shape[0]
            for i in range(1, _n):
                logdet += xp.sum(
                    xp.log(self.pobtars["A_diagonal_blocks"][i].diagonal())
                )

            if sparsity == "bta":
                logdet += xp.sum(xp.log(self.pobtars["A_arrow_tip_block"].diagonal()))
        else:
            for i in range(1, self.n_locals[self.rank] - 1):
                logdet += xp.sum(xp.log(self.A_diagonal_blocks[i].diagonal()))

        logdet = allreduce(
            logdet,
            op="sum",
            comm=self.comm,
        )

        return 2 * logdet

    def selected_inversion(
        self,
        sparsity: str,
    ) -> None:
        """Compute selected inversion of input matrix using Cholesky factor."""
        if sparsity == "bta":
            ppobtasi(
                L_diagonal_blocks=self.A_diagonal_blocks,
                L_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                L_lower_arrow_blocks=self.A_arrow_bottom_blocks,
                L_arrow_tip_block=self.A_arrow_tip_block,
                buffer=self.buffer,
                pobtars=self.pobtars,
                comm=self.comm,
                strategy="allgather",
            )
        elif sparsity == "bt":
            ppobtsi(
                L_diagonal_blocks=self.A_diagonal_blocks,
                L_lower_diagonal_blocks=self.A_lower_diagonal_blocks,
                buffer=self.buffer,
                pobtrs=self.pobtars,
                comm=self.comm,
                strategy="allgather",
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

        n_idx = xp.array([0] + self.n_locals)
        start_idx = int(xp.cumsum(n_idx)[self.rank])
        end_idx = int(xp.cumsum(n_idx)[self.rank + 1])
        for i_A in range(start_idx, end_idx):
            i_S = i_A - start_idx
            csc_slice = A_csc[
                i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
            ]

            self.A_diagonal_blocks[i_S] = csc_slice.todense()

            if i_A < self.n_diag_blocks - 1:
                csc_slice = A_csc[
                    (i_A + 1)
                    * self.diagonal_blocksize : (i_A + 2)
                    * self.diagonal_blocksize,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ]

                self.A_lower_diagonal_blocks[i_S] = csc_slice.todense()

            if sparsity == "bta":
                csc_slice = A_csc[
                    -self.arrowhead_blocksize :,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ]

                self.A_arrow_bottom_blocks[i_S] = csc_slice.todense()

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

        n_idx = xp.array([0] + self.n_locals)
        start_idx = int(xp.cumsum(n_idx)[self.rank])
        end_idx = int(xp.cumsum(n_idx)[self.rank + 1])
        for i_A in range(start_idx, end_idx):
            i_S = i_A - start_idx
            # Extract the sparsity pattern of the current Diagonal, Lower, and Arrowhead blocks
            B_coo = B[
                i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
            ].tocoo()
            row_diag, col_diag = B_coo.row, B_coo.col
            B[
                i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
            ] = sp.sparse.coo_matrix(
                (self.A_diagonal_blocks[i_S, row_diag, col_diag], (row_diag, col_diag)),
                shape=B[
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ].shape,
            )

            if i_A < self.n_diag_blocks - 1:
                B_coo = B[
                    (i_A + 1)
                    * self.diagonal_blocksize : (i_A + 2)
                    * self.diagonal_blocksize,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ].tocoo()
                row_lower, col_lower = B_coo.row, B_coo.col
                B[
                    (i_A + 1)
                    * self.diagonal_blocksize : (i_A + 2)
                    * self.diagonal_blocksize,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ] = sp.sparse.coo_matrix(
                    (
                        self.A_lower_diagonal_blocks[i_S, row_lower, col_lower],
                        (row_lower, col_lower),
                    ),
                    shape=B[
                        (i_A + 1)
                        * self.diagonal_blocksize : (i_A + 2)
                        * self.diagonal_blocksize,
                        i_A
                        * self.diagonal_blocksize : (i_A + 1)
                        * self.diagonal_blocksize,
                    ].shape,
                )

            if sparsity == "bta":
                B_coo = B[
                    -self.arrowhead_blocksize :,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ].tocoo()
                row_arrow, col_arrow = B_coo.row, B_coo.col
                B[
                    -self.arrowhead_blocksize :,
                    i_A * self.diagonal_blocksize : (i_A + 1) * self.diagonal_blocksize,
                ] = sp.sparse.coo_matrix(
                    (
                        self.A_arrow_bottom_blocks[i_S, row_arrow, col_arrow],
                        (row_arrow, col_arrow),
                    ),
                    shape=B[
                        -self.arrowhead_blocksize :,
                        i_A
                        * self.diagonal_blocksize : (i_A + 1)
                        * self.diagonal_blocksize,
                    ].shape,
                )

        if sparsity == "bta":
            B_coo = B[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :].tocoo()
            row_arrow_tip, col_arrow_tip = B_coo.row, B_coo.col
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

        # TODO: Need to communicate to agregates/Map the local B matrix to all ranks
        # Need to operate on the datas

        # Symmetrize B
        B = B + sp.sparse.tril(B, k=-1).T

        return B

    def _slice_rhs(
        self,
        rhs: NDArray,
        sparsity: str,
    ) -> NDArray:
        """Slice the right-hand side vector."""
        n_idx = xp.array([0] + self.n_locals)
        start_idx = int(xp.cumsum(n_idx)[self.rank])
        end_idx = int(xp.cumsum(n_idx)[self.rank + 1])

        # Ensure rhs is a 2D array with shape (n, 1)
        if rhs.ndim == 1:
            rhs = rhs[:, None]

        self.B[: -self.arrowhead_blocksize] = rhs[
            start_idx * self.diagonal_blocksize : end_idx * self.diagonal_blocksize
        ]
        if sparsity == "bta":
            self.B[-self.arrowhead_blocksize :] = rhs[-self.arrowhead_blocksize :, :]

    def _gather_rhs(
        self,
        rhs: xp.ndarray,
        sparsity: str,
    ):
        """Gather the right-hand side vector."""
        if rhs.ndim == 1:
            rhs = rhs[:, None]

        # Calculate the start and end indices for the local slice of rhs
        n_idx = xp.array([0] + self.n_locals)
        start_idx = int(xp.cumsum(n_idx)[self.rank])
        end_idx = int(xp.cumsum(n_idx)[self.rank + 1])

        # 1. Map back the local result of self.B in the global rhs
        rhs[start_idx * self.diagonal_blocksize : end_idx * self.diagonal_blocksize] = (
            self.B[: -self.arrowhead_blocksize]
        )
        if sparsity == "bta":
            rhs[-self.arrowhead_blocksize :] = self.B[-self.arrowhead_blocksize :]

        # 2. Communicate the rhs, AllGatherV on the global rhs
        recv_counts = xp.array(self.n_locals) * self.diagonal_blocksize
        displacements = xp.cumsum(recv_counts) - recv_counts

        allgatherv(
            sendbuf=self.B[: -self.arrowhead_blocksize],
            recvbuf=rhs[: -self.arrowhead_blocksize],
            recv_counts=recv_counts,
            displacements=displacements,
            comm=self.comm,
        )
