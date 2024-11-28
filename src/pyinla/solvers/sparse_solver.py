# Copyright 2024 pyINLA authors. All rights reserved.

import time

from pyinla import xp, sp, ArrayLike

from pyinla import ArrayLike
from pyinla.core.pyinla_config import SolverConfig
from pyinla.core.solver import Solver


class SparseSolver(Solver):
    """Abstract core class for numerical solvers."""

    def __init__(
        self,
        solver_config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(solver_config)

        self.L: sp.sparse.sparray = None
        self.A_inv: ArrayLike = None

        # store non-zero indices of A
        self.rows = None
        self.cols = None

    def cholesky(self, A: sp.sparse.sparray, **kwargs) -> None:
        """Compute Cholesky factor of input matrix."""

        A = sp.csc_matrix(A)

        if self.rows is None:
            self.rows, self.cols = A.nonzero()

        # print("Calling ScipySolver.cholesky now.")

        t_chol = time.time()
        LU = sp.sparse.linalg.splu(A, diag_pivot_thresh=0, permc_spec="NATURAL")
        t_chol = time.time() - t_chol

        if (LU.U.diagonal() > 0).all():  # Check the matrix A is positive definite.
            self.L = LU.L.dot(sp.sparse.diags(LU.U.diagonal() ** 0.5))
        else:
            raise ValueError("The matrix is not positive definite")

    def solve(
        self,
        rhs: ArrayLike,
        **kwargs,
    ) -> ArrayLike:
        """Solve linear system using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        y = sp.sparse.linalg.spsolve_triangular(self.L, rhs, lower=True)
        x = sp.sparse.linalg.spsolve_triangular(self.L.T, y, lower=False)

        return x

    def logdet(
        self,
        **kwargs,
    ) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        return 2 * xp.sum(xp.log(self.L.diagonal()))
