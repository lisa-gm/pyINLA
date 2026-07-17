# src/dalia/backend/linalg/solvers/sparse/sparse_linear_solver.py


import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu as h_splu

from dalia.backend.config import cupy_version, target_list
from dalia.backend.linalg.solvers.linear_solver import LinearSolver

if cupy_version is not None:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import splu as a_splu


class SparseSolver(LinearSolver):
    ...

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
    def _compute_factorization(self, overwrite: bool = False):
        """Compute Cholesky factorization: A = LL^T.

        Uses scipy.linalg.cholesky with LAPACK backend.

        Parameters
        ----------
        overwrite : bool
            If True, overwrites self._matrix._data with L factor.

        Returns
        -------
        L : superLU object
             SuperLU factorization object containing L and U factors.
        """
        # TODO: maybe just say that there is no cholesky decomposition
        # pylint: disable=protected-access
        if self._target == "host":
            factors = h_splu(self._matrix._data.tocsc())
        elif self._target == "accelerator" and cupy_version is not None:
            factors = a_splu(self._matrix._data.tocsc())
        else:
            raise ValueError(
                f"Invalid hardware target type '{self._target}'. Supported target types are {target_list}."
            )

        return factors

    def _solve_system(self, b: np.ndarray):
        """Solve Ax = b using LU factors.

        Solves A * x = b.

        Parameters
        ----------
        b : numpy.ndarray
            Right-hand side vector or matrix.

        Returns
        -------
        x : numpy.ndarray
            Solution vector or matrix.
        """
        # Solve A * x = b
        x = self._factors.solve(b)

        return x

    def _compute_logdet(self):
        """Compute log-determinant from LU factors.

        For LU decomposition:
            det(A) = det(L) * det(U)
            Since L has 1s on diagonal: det(L) = 1
            Therfore det(A) = det(U)
        Returns
        -------
        float
            Log-determinant of the matrix.
        """
        if self._target == "host":
            return np.sum(np.log(np.abs(self._factors.U.diagonal())))
        elif self._target == "accelerator" and cupy_version is not None:
            return cp.sum(cp.log(cp.abs(self._factors.U.diagonal())))
        else:
            raise ValueError(
                f"Invalid hardware target type '{self._target}'. Supported target types are {target_list}."
            )

    def _compute_selected_inverse(self):

        raise NotImplementedError(
            "Log determinant not implemented for sparse solver yet."
        )
