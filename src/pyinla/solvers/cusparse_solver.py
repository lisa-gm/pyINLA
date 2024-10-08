# Copyright 2024 pyINLA authors. All rights reserved.

import time

# import numpy as np
from numpy.typing import ArrayLike

try:
    import cupy as cp
    from cupyx.scipy.sparse import csc_matrix, diags, spmatrix
    from cupyx.scipy.sparse.linalg import splu, spsolve_triangular
except ImportError:
    print("CuPy is required for CuSparseSolver.")
    from scipy.sparse import csc_matrix, diags, spmatrix
    from scipy.sparse.linalg import splu, spsolve_triangular

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.core.solver import Solver


class CuSparseSolver(Solver):
    """Abstract core class for numerical solvers."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the solver."""
        super().__init__(pyinla_config)

        self.L: spmatrix = None
        self.A_inv: ArrayLike = None

        # store non-zero indices of A
        self.rows = None
        self.cols = None

    def cholesky(self, A: spmatrix) -> None:
        """Compute Cholesky factor of input matrix."""

        if self.rows is None:
            rows, cols = A.nonzero()

            self.rows = cp.array(rows)
            self.cols = cp.array(cols)

        A_device = csc_matrix(A)

        tic = time.perf_counter()
        LU = splu(A_device, diag_pivot_thresh=0, permc_spec="NATURAL")
        toc = time.perf_counter()
        print(f"Cholesky factorization took {toc - tic:0.4f} seconds")

        if (LU.U.diagonal() > 0).all():  # Check the matrix A is positive definite.
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

        # H2D transfer
        rhs_device = cp.array(rhs)

        y = spsolve_triangular(self.L, rhs_device, lower=True)
        x = spsolve_triangular(self.L.T, y, lower=False)

        # D2H transfer
        x_host = cp.asnumpy(x)

        return x_host

    def logdet(self) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        logdet_device = 2 * cp.sum(cp.log(self.L.diagonal()))
        logdet_host = cp.asnumpy(logdet_device)

        return logdet_host

    def full_inverse(self) -> None:
        """Compute inverse of input matrix using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        L_inv = splu(self.L, diag_pivot_thresh=0, permc_spec="NATURAL").solve(
            cp.eye(self.L.shape[0])
        )
        self.A_inv = L_inv.T @ L_inv

    def get_selected_inverse(self) -> spmatrix:
        """
        Create a sparse matrix A_inv_selected which contains all the entries of A_inv
        at locations where A is non-zero, and zero otherwise.
        NOTE: Could be optimized to only contain lower triangular part of A_inv
        if A is symmetric.

        Parameters
        ----------
        A_inv : cp.ndarray
            Full inverse of matrix A (dense array).

        Returns
        -------
        A_inv_selected : scipy.sparse.spmatrix
            Sparse matrix containing entries of A_inv at locations where A is non-zero.
        """

        # Extract the values of A_inv at these non-zero positions
        values = self.A_inv[self.rows, self.cols]

        # Create the sparse matrix with the same non-zero structure as A
        A_inv_selected = csc_matrix(
            (values, (self.rows, self.cols)), shape=self.L.shape
        )

        return A_inv_selected.get()

    def selected_inverse(self) -> spmatrix:
        """Compute inverse of nonzero sparsity pattern of L."""

        raise NotImplementedError
