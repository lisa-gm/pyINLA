# Copyright 2024-2025 DALIA authors. All rights reserved.

import time

from dalia import NDArray, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver
from dalia.utils import synchronize_gpu

# This is a workaround a problem in cupyx, where linalg is not properly namespaced (directly accessible).
# May be removed in future versions of cupy (tested on cupy 13.4.1).
if xp.__name__ == "cupy":
    from cupyx.scipy.sparse.linalg import splu
else:
    from scipy.sparse.linalg import splu


class SparseSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config)

        self.LU_factor = None  # Store the LU factorization object

        # Solver Metrics
        self.t_factorize = 0.0
        self.t_solve = 0.0

    def factorize(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        """Compute the decomposition of a matrix.

        Note: This uses LU decomposition since sparse Cholesky is not readily available.


        Parameters
        ----------
        A : sp.sparse.spmatrix
            The input matrix to decompose.

        Returns
        -------
        None

        Note:
        -----
        Uses the LU decomposition by default as scipy.sparse doesn't implement Cholesky.
        """
        synchronize_gpu()
        tic = time.perf_counter()

        A = sp.sparse.csc_matrix(A)

        # Use LU decomposition as the factorization method
        self.LU_factor = splu(A, diag_pivot_thresh=0, permc_spec="NATURAL")

        # Check if the matrix appears to be positive definite
        if not (self.LU_factor.U.diagonal() > 0).all():
            raise ValueError("The matrix does not appear to be positive definite")

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_factorize += toc - tic

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using LU factorization.

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

        if self.LU_factor is None:
            raise ValueError("Matrix factorization not computed")

        # Handle multiple RHS cases
        if rhs.ndim == 1:
            # Single RHS as 1D array
            x = self.LU_factor.solve(rhs)
        elif rhs.ndim == 2 and rhs.shape[1] == 1:
            # Single RHS as column vector
            x = self.LU_factor.solve(rhs.flatten())
            x = x.reshape(rhs.shape)
        elif rhs.ndim == 2 and rhs.shape[1] > 1:
            # Multiple RHS (batched) - scipy splu can handle this directly
            x = self.LU_factor.solve(rhs)
        else:
            raise ValueError(f"Unsupported RHS shape: {rhs.shape}")

        synchronize_gpu()
        toc = time.perf_counter()
        self.t_solve += toc - tic

        return x

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

        if self.LU_factor is None:
            raise ValueError("Matrix factorization not computed")

        # For LU decomposition: det(A) = det(L) * det(U)
        # Since L has 1s on diagonal: det(L) = 1
        # So det(A) = det(U) = product of diagonal elements of U
        log_det_U = xp.sum(xp.log(xp.abs(self.LU_factor.U.diagonal())))

        return float(log_det_U)

    def selected_inversion(self, **kwargs):
        """Compute selected inversion of input matrix using LU factorization.

        Raises:
        ------
        NotImplementedError
            Selected inversion is not implemented for SparseSolver.
        """
        raise NotImplementedError(
            "Selected inversion is not implemented for SparseSolver."
        )

    def _structured_to_spmatrix(self, **kwargs) -> None:
        """Convert structured matrix to sparse matrix.

        For SparseSolver, this is a no-op since it works directly with sparse matrices.
        """
        pass

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""
        if self.LU_factor is None:
            return 0

        # Estimate memory usage from L and U matrices
        L_memory = (
            self.LU_factor.L.data.nbytes
            + self.LU_factor.L.indptr.nbytes
            + self.LU_factor.L.indices.nbytes
        )
        U_memory = (
            self.LU_factor.U.data.nbytes
            + self.LU_factor.U.indptr.nbytes
            + self.LU_factor.U.indices.nbytes
        )
        return L_memory + U_memory
