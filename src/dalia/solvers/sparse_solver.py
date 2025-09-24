# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import NDArray, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver


class SparseSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config)

        self.L: sp.sparse.spmatrix = None

    def cholesky(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        """Compute Cholesky factor of input matrix."""

        A = sp.sparse.csc_matrix(A)

        LU = sp.sparse.linalg.splu(A, diag_pivot_thresh=0, permc_spec="NATURAL")

        if (LU.U.diagonal() > 0).all():  # Check the matrix A is positive definite.
            self.L = LU.L.dot(sp.sparse.diags(LU.U.diagonal() ** 0.5))
        else:
            # print("min(diag(L)): ", xp.min(LU.U.diagonal()))
            raise ValueError("The matrix is not positive definite")

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        rhs[:] = sp.sparse.linalg.spsolve_triangular(
            self.L, rhs, lower=True, overwrite_b=True
        )
        rhs[:] = sp.sparse.linalg.spsolve_triangular(
            self.L.T, rhs, lower=False, overwrite_b=True
        )

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        return 2 * xp.sum(xp.log(self.L.diagonal()))

    def selected_inversion(self, **kwargs) -> NDArray:
        # convert to dense
        L_dense = self.L.toarray()
        L_inv = xp.eye(self.L.shape[0])

        L_inv[:] = sp.linalg.solve_triangular(
            L_dense, L_inv, lower=True, overwrite_b=True
        )
        self.A_inv = L_inv.T @ L_inv

        return self.A_inv

    def _structured_to_spmatrix(self, A: sp.sparse.spmatrix, **kwargs) -> NDArray:
        B = A.tocoo()
        B.data = self.A_inv[B.row, B.col]

        return B

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""
        if self.L is None:
            return 0

        return self.L.data.nbytes + self.L.indptr.nbytes + self.L.indices.nbytes
