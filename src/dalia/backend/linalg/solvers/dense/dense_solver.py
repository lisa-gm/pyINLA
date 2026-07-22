# src/dalia/backend/linalg/solvers/dense/dense_linear_solver.py

from dalia.backend.config import cupy_version, nvmath_version, target_list
import numpy as np
import scipy.linalg as sp_la
# from scipy.linalg import cholesky, get_lapack_funcs, solve_triangular


if cupy_version is not None:
    import cupy as cp
    import cupy.linalg as cp_la

if nvmath_version is not None:
    from nvmath.bindings import cublas as nvcublas

from dalia.backend.linalg.solvers.linear_solver import LinearSolver
from dalia.backend.BLAS import trsm, gemm

class DenseSolver(LinearSolver):
    """Base Dense linear solver class.

    Relying on scipy.linalg for dense linear algebra operations.
    """

    # 1. Class attributes (if any)
    # 2. Initialization
    # 3. Special representation methods
    # 4. Properties (grouped together)
    # 5. Comparison operators (if needed)
    # 6. Arithmetic operators (standard order)
    # 7. Right-hand operators (same order as above)
    # 8. In-place operators (if supported)
    # 9. Other special methods
    # 10. Public methods
    # 11. Private/protected methods (start with _)
    def _compute_factorization(self, overwrite: bool = False) -> np.ndarray:
        """Compute Cholesky factorization: A = LL^T.

        Uses scipy.linalg.cholesky with LAPACK backend.

        Parameters
        ----------
        overwrite : bool
            If True, overwrites self._matrix._data with L factor.

        Returns
        -------
        L : numpy.ndarray
            Lower triangular Cholesky factor.
        """
        # TODO: maybe use pbstf instead
        if self._target == "host":
            factors = sp_la.cholesky(
                self._matrix._data, lower=True, overwrite_a=overwrite, check_finite=False
            )
        elif self._target == "accelerator":
            factors = cp_la.cholesky(
                self._matrix._data
            )
        else:
            raise ValueError(f"Unsupported hardware target: {self._target}")
        # pylint: disable=protected-access
        return factors

    def _solve_system(self, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b using Cholesky factors.

        Solves LL^T x = b via forward and backward substitution.

        Parameters
        ----------
        b : numpy.ndarray
            Right-hand side vector or matrix.

        Returns
        -------
        x : numpy.ndarray
            Solution vector or matrix.
        """
        if self._target == "accelerator":
            b = cp.asarray(b)
        # Forward solve L y = b
        y = trsm(
            self._factors,
            b,
            hw_target=self._target,
            lower=True,
            overwrite_b=False,
            check_finite=False,
        )

        # Backward solve L^T x = y using the transpose of the lower-triangular factor
        x = trsm(
            self._factors,
            y,
            hw_target=self._target,
            trans="T",
            lower=True,
            overwrite_b=False,
            check_finite=False,
        )

        return x

    def _compute_logdet(self) -> float:
        """Compute log-determinant from Cholesky factors.

        For Cholesky decomposition A = LL^T:
            log|A| = log|LL^T| = log|L|^2 = 2*log|L|
            log|L| = sum(log(diag(L)))  (triangular matrix)

        Returns
        -------
        float
            Log-determinant of the matrix.
        """
        if self._target == "host":
            return 2.0 * np.sum(np.log(np.diag(self._factors)))
        elif self._target == "accelerator":
            return 2.0 * cp.sum(cp.log(cp.diag(self._factors)))
        else:
            raise ValueError(f"Invalid hardware target type '{self._target}'. Supported target types are {target_list}.")

    def _compute_selected_inverse(self, overwrite_factors: bool = False):
        """Compute full matrix inverse as DenseMatrix.

        For Cholesky decomposition A = LL^T, computes A^{-1}.

        Parameters
        ----------
        overwrite_factors : bool
            If True, computes inverse in-place over self._factors (saves memory).
            If False, preserves factors and allocates new memory for inverse.

        Returns
        -------
        DenseMatrix
            Full matrix inverse. When A is a precision matrix, this is the
            covariance matrix.

        Notes
        -----
        Memory usage:
        - overwrite_factors=False: Allocates 2xn² additional memory (L_inv + result)
        - overwrite_factors=True: Allocates 0 additional memory (in-place via POTRI)

        Uses LAPACK's POTRI routine which computes the inverse in-place by:
        1. Inverting triangular factor L → L^{-1} (via TRTRI)
        2. Computing A^{-1} = L^{-1}^T @ L^{-1} (via LAUUM)
        3. Storing result in lower triangle only (then symmetrized)

        Warnings
        --------
        When overwrite_factors=True, self._factors is destroyed and contains
        the inverse afterward. The solver becomes unusable until refactorization.
        """
        # pylint: disable=invalid-name,import-outside-toplevel
        from dalia.backend.datastructures import DenseMatrix

        n = self._factors.shape[0]
        if self._target == "host":
            if overwrite_factors:
                # Avoid extra copies and fully work in-place.
                #   - This uses LAPACK POTRI to directly inverse the L factor, after
                #   the call, self._factors contains the inverse in its lower triangle.
                #   - As POTRI only fills the lower triangle, we symmetrize afterward.
                
                (potri,) = sp_la.get_lapack_funcs(("potri",), (self._factors,))

                inv_array, info = potri(self._factors, lower=True, overwrite_c=True)

                if info != 0:
                    raise np.linalg.LinAlgError(f"POTRI failed with error code {info}")

                i_lower = np.tril_indices(n, -1)
                inv_array[i_lower[::-1]] = inv_array[i_lower]

            else:
                # Safe: compute in new memory, preserving factors
                # This allocates 2xn² additional memory at peak

                # Compute L^{-1} by solving L X = I
                L_inv = sp_la.solve_triangular(
                    self._factors, np.eye(n), lower=True, check_finite=False
                )

                # Compute A^{-1} = L_inv^T @ L_inv
                inv_array = L_inv.T @ L_inv
        elif self._target == "accelerator":
            if overwrite_factors and nvmath_version is not None:
                # Avoid extra copies and fully work in-place.
                #   - This uses LAPACK POTRI to directly inverse the L factor, after
                #   the call, self._factors contains the inverse in its lower triangle.
                #   - As POTRI only fills the lower triangle, we symmetrize afterward.

                dtype = self._factors.dtype.char
                if dtype == 'f':
                    func = nvcublas.spotri
                elif dtype == 'd':
                    func = nvcublas.dpotri
                elif dtype == 'F':
                    func = nvcublas.cpotri
                elif dtype == 'D':
                    func = nvcublas.zpotri
                else:
                    raise TypeError('invalid dtype')
                
                inv_array, info = potri(self._factors, lower=True, overwrite_c=True)

                if info != 0:
                    raise cp.linalg.LinAlgError(f"POTRI failed with error code {info}")
                
                i_lower = cp.tril_indices(n, -1)
                inv_array[i_lower[::-1]] = inv_array[i_lower]

            else:
                # Safe: compute in new memory, preserving factors
                # This allocates 2xn² additional memory at peak

                # Compute L^{-1} by solving L X = I
                L_inv = trsm(
                    self._factors, cp.eye(n), hw_target=self._target, lower=True, check_finite=False
                )

                # Compute A^{-1} = L_inv^T @ L_inv
                #inv_array = L_inv.T @ L_inv
                inv_array = gemm(L_inv, L_inv, hw_target=self._target, trans_a="T")
        else:
            raise ValueError(f"Invalid hardware target type '{self._target}'. Supported target types are {target_list}.")
        return DenseMatrix(inv_array)
    
