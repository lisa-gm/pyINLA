# tests/backend/units/datastructures/matrix/conftest.py

import numpy as np
import scipy.sparse as sp

import pytest

from dalia.backend.datastructures import SparseMatrix, DenseMatrix

# Type groups - reusable across all tests
EXTERNAL_SPARSE_TYPES = ["scipy_csr", "scipy_csc", "scipy_coo"]
EXTERNAL_DENSE_TYPES = ["numpy"]


@pytest.fixture
def matrix_factory():
    """Factory to create test matrices of different types"""

    def _make_matrix(matrix_type, shape=(3, 3), data=None):
        if data is None:
            data = np.arange(1, shape[0] * shape[1] + 1).reshape(shape)

        # Handle Internal types
        if matrix_type == "SparseMatrix":
            return SparseMatrix(sp.csr_matrix(data))
        if matrix_type == "DenseMatrix":
            return DenseMatrix(data)

        # Handle External types
        if matrix_type == "scipy_csr":
            return sp.csr_array(data)
        if matrix_type == "scipy_csc":
            return sp.csc_array(data)
        if matrix_type == "scipy_coo":
            return sp.coo_array(data)
        if matrix_type == "numpy":
            return data

        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    return _make_matrix
