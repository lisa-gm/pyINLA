# Copyright 2024 pyINLA authors. All rights reserved.

import time

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix, diags, sparray
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
        self.A_inv: ArrayLike = None

        # store non-zero indices of A
        self.non_zero_rows = None
        self.non_zero_cols = None

    def cholesky(self, A: sparray) -> None:
        """Compute Cholesky factor of input matrix."""

        A = csc_matrix(A)

        if self.non_zero_rows is None:
            self.non_zero_rows, self.non_zero_cols = A.nonzero()

        # print("Calling ScipySolver.cholesky now.")

        t_chol = time.time()
        LU = splu(A, diag_pivot_thresh=0, permc_spec="NATURAL")
        t_chol = time.time() - t_chol

        if (LU.U.diagonal() > 0).all():  # Check the matrix A is positive definite.
            self.L = LU.L.dot(diags(LU.U.diagonal() ** 0.5))
        else:
            raise ValueError("The matrix is not positive definite")

        # print("ScipySolver.cholesky done. Time taken: ", t_chol)

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

    def full_inverse(self) -> None:
        """Compute inverse of input matrix using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        L_inv = splu(self.L, diag_pivot_thresh=0, permc_spec="NATURAL").solve(
            np.eye(self.L.shape[0])
        )
        self.A_inv = L_inv.T @ L_inv

    def extract_selected_inverse(self, A_inv: ArrayLike) -> sparray:
        """
        Create a sparse matrix A_inv_selected which contains all the entries of A_inv
        at locations where A is non-zero, and zero otherwise.
        NOTE: Could be optimized to only contain lower triangular part of A_inv if A is symmetric.

        Parameters
        ----------
        A_inv : np.ndarray
            Full inverse of matrix A (dense array).

        Returns
        -------
        A_inv_selected : scipy.sparse.spmatrix
            Sparse matrix containing entries of A_inv at locations where A is non-zero.
        """

        # Extract the values of A_inv at these non-zero positions
        values = A_inv[self.non_zero_rows, self.non_zero_cols]

        # Create the sparse matrix with the same non-zero structure as A
        A_inv_selected = csc_matrix(
            (values, (self.non_zero_rows, self.non_zero_cols)), shape=self.L.shape
        )

        return A_inv_selected

    def selected_inverse(self) -> sparray:
        """Compute inverse of nonzero sparsity pattern of L."""

        raise NotImplementedError
