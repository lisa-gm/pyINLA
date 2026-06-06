# src/dalia/backend/linalg/solvers/sparse/sparse_linear_solver.py



import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu

from dalia.backend.linalg.solvers.linear_solver import LinearSolver
from dalia.backend.config import cupy_version

if cupy_version is not None:
    import cupy as cp

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
        if self._target == "accelerator":
            raise NotImplementedError("SparseSolver does not support accelerators. Use CuDSS instead.")
        factors = splu(
            self._matrix._data.tocsc()
        )
        return factors
    
    def _solve_system(self, b: np.ndarray):
        """Solve Ax = b using Cholesky factors.

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
        if self._target == "accelerator":
            raise NotImplementedError("SparseSolver does not support accelerators. Use CuDSS instead.")
        # Forward solve L y = b

        # Backward solve L^T x = y
        x = self._factors.solve(b
        )

        return x
    
    def _compute_selected_inverse(self):

        return NotImplementedError("Selected inversion not implemented for sparse solver yet.")
    
    def _compute_logdet(self):

        return NotImplementedError("Log determinant not implemented for sparse solver yet.")