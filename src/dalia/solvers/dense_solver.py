# Copyright 2024-2025 DALIA authors. All rights reserved.

import time

from dalia import NDArray, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver
from dalia.utils import synchronize_gpu


## check if sparse matrix is diagonal
def is_diagonal(A: NDArray) -> bool:
    """Check if a matrix is diagonal."""

    coo = A.tocoo()
    return xp.all(coo.row == coo.col)


class DenseSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the DenseSolver class.

        Parameters
        ----------
        config : SolverConfig
            Configuration object for the solver.

        Returns
        -------
        None
        """
        super().__init__(config)

        self.n: int = kwargs.get("n", None)
        assert self.n is not None, "The size of the matrix must be provided."

        # self.L: NDArray = xp.zeros((self.n, self.n), dtype=xp.float64)
        self.L = None
        self.A_inv = None

        # Solver Metrics
        self.t_cholesky = 0.0
        self.t_solve = 0.0

    def cholesky(self, A: NDArray, **kwargs) -> None:
        """Compute the Cholesky decomposition of a matrix.

        Parameters
        ----------
        A : NDArray
            The input matrix to decompose.

        Returns
        -------
        None
        """
        synchronize_gpu()
        tic = time.perf_counter()

        if sp.sparse.issparse(A):
            # if A is diagonal, we can use the diagonal directly
            if is_diagonal(A):
                print(f"{self.n=}")
                print(f"{self.L is None=}")
                if self.L is None:
                    self.L = xp.zeros((self.n, self.n), dtype=xp.float64)
                else:
                    self.L[:] = 0
                print(f"{self.L.shape=}")
                # self.L.diagonal()[:] = xp.sqrt(A.diagonal())
                d_A = A.diagonal()
                print(f"{xp.min(d_A)=}, {xp.max(d_A)=}, {xp.mean(d_A)=}")
                # self.L[xp.arange(self.n), xp.arange(self.n)] = xp.sqrt(A.diagonal())
                xp.fill_diagonal(self.L, xp.sqrt(A.diagonal()))
                d_L = xp.diag(self.L)
                print(f"{xp.min(d_L)=}, {xp.max(d_L)=}, {xp.mean(d_L)=}")
                return

            else:
                # self.L[:] = A.todense()
                tmp = A.todense()
        else:
            # self.L[:] = A
            tmp = A

        self.L = xp.linalg.cholesky(tmp)

        d_tmp = xp.diag(tmp)
        print(f"{xp.min(d_tmp)=}, {xp.max(d_tmp)=}, {xp.mean(d_tmp)=}")
        d_L = xp.diag(self.L)
        print(f"{xp.min(d_L)=}, {xp.max(d_L)=}, {xp.mean(d_L)=}")

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_cholesky += toc - tic

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using Cholesky factor.

        Parameters
        ----------
        rhs : NDArray
            Right-hand side of the linear system.

        Returns
        -------
        NDArray
            Solution of the linear system.
        """
        synchronize_gpu()
        tic = time.perf_counter()

        rhs[:] = sp.linalg.solve_triangular(self.L, rhs, lower=True, overwrite_b=True)
        rhs[:] = sp.linalg.solve_triangular(
            self.L.T, rhs, lower=False, overwrite_b=True
        )

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_solve += toc - tic

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        """Compute the log determinant of the matrix.

        Returns
        -------
        float
            The log determinant of the matrix.
        """
        if xp.isnan(self.L).any() or xp.isinf(self.L).any():
            raise ValueError("Cholesky factor L is NaN or Inf. Check what is happening.")
        d = xp.diag(self.L)
        print(f"{xp.min(d)=}, {xp.max(d)=}, {xp.mean(d)=}")
        res = 2 * xp.sum(xp.log(xp.diag(self.L)))
        if xp.isnan(res) or xp.isinf(res):
            raise ValueError("Log determinant is NaN or Inf. Check what is happening.")
        return res

    def selected_inversion(self, **kwargs) -> None:
        L_inv = xp.eye(self.L.shape[0])
        L_inv[:] = sp.linalg.solve_triangular(
            self.L, L_inv, lower=True, overwrite_b=True
        )
        self.A_inv = L_inv.T @ L_inv

        return self.A_inv

    def _structured_to_spmatrix(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        B = A.tocoo()
        B.data = self.A_inv[B.row, B.col]

        return B

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes."""
        solver_mem = 2 * self.n * self.n * xp.dtype(xp.float64).itemsize

        return solver_mem
