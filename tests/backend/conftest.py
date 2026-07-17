

import numpy as np
import pytest
import scipy.sparse as sp

from dalia.backend.datastructures import DenseMatrix, SparseMatrix
from dalia.backend.config import cupy_version

if cupy_version is not None:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp

@pytest.fixture
def matrix_factory():
    """Factory to create test matrices of different types"""

    def _make_matrix(matrix_type, shape=(3, 3), data=None, hw_target=None):
        if data is None:
            data = np.arange(1, shape[0] * shape[1] + 1).reshape(shape)
            data = data.astype(np.float64)

        # Handle Internal types
        if matrix_type == "SparseMatrix":
            return SparseMatrix(sp.csr_matrix(data, dtype=np.float64), hw_target=hw_target)
        if matrix_type == "DenseMatrix":
            return DenseMatrix(data, hw_target=hw_target)

        # Handle External types
        # . numpy/scipy
        if matrix_type == "scipy_csr":
            return sp.csr_array(data, dtype=np.float64)
        if matrix_type == "scipy_csc":
            return sp.csc_array(data, dtype=np.float64)
        if matrix_type == "scipy_coo":
            return sp.coo_array(data, dtype=np.float64)
        if matrix_type == "numpy":
            return data
        
        # . cupy/cupy-sparse
        if matrix_type == "cupy_csr":
            data = sp.csr_matrix(data)
            return cu_sp.csr_matrix(data, dtype = cp.float64)
        if matrix_type == "cupy_csc":
            data = sp.csc_matrix(data)
            return cu_sp.csc_matrix(data, dtype = cp.float64)
        if matrix_type == "cupy_coo":
            data = sp.coo_matrix(data)
            return cu_sp.coo_matrix(data, dtype = cp.float64)
        if matrix_type == "cupy":
            return cp.asarray(data, dtype=cp.float64)

        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    return _make_matrix