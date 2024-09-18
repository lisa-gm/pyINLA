# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import diags, sparray
from scipy.sparse.linalg import splu, spsolve_triangular

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.core.solver import Solver


class ScipySolver(Solver):
    """Abstract core class for numerical solvers."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the solver."""
        super().__init__(pyinla_config)

        self.L: sparray = None

    def cholesky(self, A: sparray) -> None:
        """Compute Cholesky factor of input matrix."""

        n = A.shape[0]
        LU = splu(A, diag_pivot_thresh=0)

        if (LU.perm_r == np.arange(n)).all() and (
            LU.U.diagonal() > 0
        ).all():  # Check the matrix A is positive definite.
            self.L = LU.L.dot(diags(LU.U.diagonal() ** 0.5))
        else:
            raise ValueError("The matrix is not positive definite")

    def solve(
        self,
        rhs: ArrayLike,
    ) -> ArrayLike:
        """Solve linear system using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        y = spsolve_triangular(self.L, rhs, lower=True)
        x = spsolve_triangular(self.L.T, y, lower=False)

        return x

    def logdet(self) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        return 2 * np.sum(np.log(self.L.diagonal()))
