# Copyright 2024 pyINLA authors. All rights reserved.

import time

import numpy as np
from cupyx.profiler import time_range
from numpy.typing import ArrayLike
from scipy.sparse import sparray

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.core.solver import Solver
from pyinla.utils.other_utils import print_mpi

try:
    from serinv import pobtaf, pobtas, pobtf
except ImportError:
    print_mpi("Serinv not installed. Please install serinv to use SerinvSolver.")


try:
    import cupy as cp
    import cupyx as cpx
except ImportError:
    print_mpi("CuPy not installed. Please install cupy to use GPU compute.")


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

        # --- Initialize memory for BTA-array storage
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
        else:
            raise NotImplementedError("Solve is only supported for BTA sparsity")

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
            L[
                -self.arrowhead_blocksize :, -self.arrowhead_blocksize :
            ] = self.A_arrow_tip_block[:, :]

        return L


class SerinvSolverGPU(Solver):
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
        self.A_diagonal_blocks = cpx.empty_pinned(
            (self.n_diagonal_blocks, self.diagonal_blocksize, self.diagonal_blocksize),
            dtype=self.dtype,
        )
        self.A_lower_diagonal_blocks = cpx.empty_pinned(
            (
                self.n_diagonal_blocks - 1,
                self.diagonal_blocksize,
                self.diagonal_blocksize,
            ),
            dtype=self.dtype,
        )
        self.A_arrow_bottom_blocks = cpx.empty_pinned(
            (self.n_diagonal_blocks, self.arrowhead_blocksize, self.diagonal_blocksize),
            dtype=self.dtype,
        )
        self.A_arrow_tip_block = cpx.empty_pinned(
            (self.arrowhead_blocksize, self.arrowhead_blocksize),
            dtype=self.dtype,
        )

        # --- Initialize device-side BTA-array storage
        self.A_diagonal_blocks_d = cp.empty(
            (self.n_diagonal_blocks, self.diagonal_blocksize, self.diagonal_blocksize),
            dtype=self.dtype,
        )
        self.A_lower_diagonal_blocks_d = cp.empty(
            (
                self.n_diagonal_blocks - 1,
                self.diagonal_blocksize,
                self.diagonal_blocksize,
            ),
            dtype=self.dtype,
        )
        self.A_arrow_bottom_blocks_d = cp.empty(
            (self.n_diagonal_blocks, self.arrowhead_blocksize, self.diagonal_blocksize),
            dtype=self.dtype,
        )
        self.A_arrow_tip_block_d = cp.empty(
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

        self.L_diagonal_blocks_d = self.A_diagonal_blocks_d
        self.L_lower_diagonal_blocks_d = self.A_lower_diagonal_blocks_d
        self.L_arrow_bottom_blocks_d = self.A_arrow_bottom_blocks_d
        self.L_arrow_tip_block_d = self.A_arrow_tip_block_d

    @time_range()
    def cholesky(self, A: sparray, sparsity: str = "bta") -> None:
        """Compute Cholesky factor of input matrix."""

        if sparsity == "bta":
            # with time_range('initializeBTAblocks', color_id=0):
            self._sparray_to_structured(A, sparsity="bta")
            self._h2d_buffers(sparsity="bta")

            tic = time.perf_counter()
            with time_range("PobtafBTA", color_id=0):
                (
                    self.L_diagonal_blocks_d,
                    self.L_lower_diagonal_blocks_d,
                    self.L_arrow_bottom_blocks_d,
                    self.L_arrow_tip_block_d,
                ) = pobtaf(
                    self.A_diagonal_blocks_d,
                    self.A_lower_diagonal_blocks_d,
                    self.A_arrow_bottom_blocks_d,
                    self.A_arrow_tip_block_d,
                )
            toc = time.perf_counter()
            print_mpi(
                "                 pobtaf Q_conditional time:", toc - tic, flush=True
            )

        elif sparsity == "bt":
            # with time_range('initializeBTblocks', color_id=0):
            self._sparray_to_structured(A, sparsity="bt")
            self._h2d_buffers(sparsity="bt")

            with time_range("pobtafBT", color_id=0):
                (
                    self.L_diagonal_blocks_d,
                    self.L_lower_diagonal_blocks_d,
                ) = pobtf(
                    self.A_diagonal_blocks_d,
                    self.A_lower_diagonal_blocks_d,
                )

            # Factorize the unconnected tip of the arrow
            self.L_arrow_tip_block_d[:, :] = cp.linalg.cholesky(
                self.A_arrow_tip_block_d
            )

    @time_range()
    def solve(self, rhs: ArrayLike, sparsity: str = "bta") -> ArrayLike:
        """Solve linear system using Cholesky factor."""

        rhs_d = cp.asarray(rhs)

        if sparsity == "bta":
            # with time_range('solve', color_id=0):
            X_d = pobtas(
                self.L_diagonal_blocks_d,
                self.L_lower_diagonal_blocks_d,
                self.L_arrow_bottom_blocks_d,
                self.L_arrow_tip_block_d,
                rhs_d,
            )

            return cp.asnumpy(X_d)
        else:
            raise NotImplementedError("Solve is only supported for BTA sparsity")

    @time_range()
    def logdet(self) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        logdet: float = 0.0

        for i in range(self.n_diagonal_blocks):
            logdet += cp.sum(cp.log(self.L_diagonal_blocks_d[i].diagonal()))

        logdet += cp.sum(cp.log(self.L_arrow_tip_block_d.diagonal()))
        logdet = 2 * logdet

        return cp.asnumpy(logdet)

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

        if sparsity == "bta":
            csr_slice = A_csr[-self.arrowhead_blocksize :, -self.arrowhead_blocksize :]

            self.A_arrow_tip_block[:, :] = csr_slice.todense()

    @time_range()
    def _h2d_buffers(self, sparsity: str) -> None:
        """Copy data from host to device."""

        self.A_diagonal_blocks_d.set(arr=self.A_diagonal_blocks)
        self.A_lower_diagonal_blocks_d.set(arr=self.A_lower_diagonal_blocks)

        if sparsity == "bta":
            self.A_arrow_bottom_blocks_d.set(arr=self.A_arrow_bottom_blocks)

        self.A_arrow_tip_block_d.set(arr=self.A_arrow_tip_block)

    @time_range()
    def _d2h_buffers(self, sparsity: str) -> None:
        """Copy data from host to device."""

        self.A_diagonal_blocks_d.get(out=self.A_diagonal_blocks)
        self.A_lower_diagonal_blocks_d.get(out=self.A_lower_diagonal_blocks)

        if sparsity == "bta":
            self.A_arrow_bottom_blocks_d.get(out=self.A_arrow_bottom_blocks)

        self.A_arrow_tip_block_d.get(out=self.A_arrow_tip_block)

    def full_inverse(self) -> ArrayLike:
        """Compute full inverse of A."""
        raise NotImplementedError

    def get_selected_inverse(self) -> sparray:
        """extract values of the inverse of A that are nonzero in A."""
        raise NotImplementedError

    def selected_inverse(self) -> sparray:
        """Compute inverse of nonzero sparsity pattern of L."""

        raise NotImplementedError

    @time_range()
    def get_L(self, sparsity: str = "bta") -> ArrayLike:
        """Get L as a dense array."""

        n = self.diagonal_blocksize * self.n_diagonal_blocks

        if sparsity == "bta":
            n += self.arrowhead_blocksize

        L = np.zeros(
            (n, n),
            dtype=self.A_diagonal_blocks.dtype,
        )

        self._d2h_buffers(sparsity)

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
            L[
                -self.arrowhead_blocksize :, -self.arrowhead_blocksize :
            ] = self.A_arrow_tip_block[:, :]

        return L
