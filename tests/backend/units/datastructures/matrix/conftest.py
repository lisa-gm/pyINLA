# tests/backend/units/datastructures/matrix/conftest.py

import numpy as np
import pytest
import scipy.sparse as sp

from dalia.backend.datastructures import DenseMatrix, SparseMatrix

# Type groups - reusable across all tests
EXTERNAL_SPARSE_TYPES = ["scipy_csr", "scipy_csc", "scipy_coo"]
EXTERNAL_DENSE_TYPES = ["numpy"]
INTERNAL_DEVICE_TYPES = ["host"]

# TODO: Change this to use flags instead of try
try:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp
    EXTERNAL_DENSE_TYPES.append("cupy")
    EXTERNAL_SPARSE_TYPES.append("cupy_csr")
    EXTERNAL_SPARSE_TYPES.append("cupy_csc")
    EXTERNAL_SPARSE_TYPES.append("cupy_coo")
    INTERNAL_DEVICE_TYPES.append("accelerator")

except ImportError:
    pass  # CuPy not available, skip GPU tests


@pytest.fixture
def matrix_factory():
    """Factory to create test matrices of different types"""

    def _make_matrix(matrix_type, shape=(3, 3), data=None, device=None):
        if data is None:
            data = np.arange(1, shape[0] * shape[1] + 1).reshape(shape)
            data.astype(float)

        # Handle Internal types
        if matrix_type == "SparseMatrix":
            return SparseMatrix(sp.csr_matrix(data, dtype=float), device=device)
        if matrix_type == "DenseMatrix":
            return DenseMatrix(data, device=device)

        # Handle External types
        if matrix_type == "scipy_csr":
            return sp.csr_array(data, dtype=float)
        if matrix_type == "scipy_csc":
            return sp.csc_array(data, dtype=float)
        if matrix_type == "scipy_coo":
            return sp.coo_array(data, dtype=float)
        if matrix_type == "numpy":
            return data
        # Cupy types
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
            return cp.asarray(data)

        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    return _make_matrix
