# src/dalia/backend/linalg/solvers/sparse/cudss.py
from .sparse_solver import SparseSolver
from dalia.backend.config import cupy_version, nvmath_version
import numpy as np

if cupy_version is not None:
    import cupy as cp
if nvmath_version is not None:
    import nvmath.sparse.advanced as nm

class CuDSS(SparseSolver):
    """CUDA Sparse Solver (cuDSS) for solving sparse linear systems.

    Uses NVIDIA's cuDSS library for GPU-accelerated sparse direct solving.
    Supports sparse matrices stored in CSR format on GPU memory.
    """

    # 1. Class attributes (if any)
    # 2. Initialization

    def __init__(self, matrix):
        if cupy_version is None:
            raise ImportError("CuPy is required for CuDSS. Please install CuPy to use this solver.")
        if nvmath_version is None:
            raise ImportError("NVMATH is required for CuDSS. Please install NVMATH to use this solver.")
        super().__init__(matrix)
        self._is_factorized = True # cuDSS does not allow for a separate factorization step

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
        
        # TODO: add warning or logging
        return None # cuDSS does not support separate factorization step, so we return None

    def _solve_system(self, b):

        if isinstance(b, np.ndarray):
            b = cp.asarray(b)
        
        x = nm.direct_solver(self._matrix._data, b)

        return x
    