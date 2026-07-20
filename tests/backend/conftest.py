from typing import Literal

import numpy as np
import pytest
import scipy.sparse as sp

from dalia.backend.config import cupy_version
from dalia.backend.datastructures import DenseMatrix, SparseMatrix

if cupy_version is not None:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp


# DATA_TYPES = ["float32", "float64", "complex64", "complex128"]
DATA_TYPES = [
    pytest.param("float64", id="float64"),
    pytest.param("complex128", id="complex128"),
]


@pytest.fixture(params=DATA_TYPES)
def data_type(
    request: pytest.FixtureRequest,
) -> Literal["float64", "complex128"]:
    """Fixture to provide data types for tests"""
    return request.param


@pytest.fixture
def matrix_factory():
    """Factory to create test matrices of different types"""

    data_types_np_mapping = {
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }
    if cupy_version is not None:
        data_types_cp_mapping = {
            "float32": cp.float32,
            "float64": cp.float64,
            "complex64": cp.complex64,
            "complex128": cp.complex128,
        }

    def _make_matrix(
        matrix_type,
        shape=(3, 3),
        data=None,
        dtype: str = "float64",
        hw_target=None,
    ):
        if data is None:
            data = np.arange(1, shape[0] * shape[1] + 1).reshape(shape)
            data = data.astype(data_types_np_mapping[dtype])

        # Handle Internal types
        if matrix_type == "SparseMatrix":
            return SparseMatrix(sp.csr_matrix(data, dtype=dtype), hw_target=hw_target)
        if matrix_type == "DenseMatrix":
            return DenseMatrix(data, hw_target=hw_target)

        # Handle External types
        # . numpy/scipy
        dtype_np = data_types_np_mapping[dtype]
        if matrix_type == "scipy_csr":
            return sp.csr_array(data, dtype=dtype_np)
        if matrix_type == "scipy_csc":
            return sp.csc_array(data, dtype=dtype_np)
        if matrix_type == "scipy_coo":
            return sp.coo_array(data, dtype=dtype_np)
        if matrix_type == "numpy":
            return data

        # . cupy/cupy-sparse
        # . convert numpy dtype to cupy dtype
        if matrix_type == "cupy_csr":
            data = sp.csr_matrix(data)
            dtype_cp = data_types_cp_mapping[dtype]
            return cu_sp.csr_matrix(data, dtype=dtype_cp)
        if matrix_type == "cupy_csc":
            data = sp.csc_matrix(data)
            dtype_cp = data_types_cp_mapping[dtype]
            return cu_sp.csc_matrix(data, dtype=dtype_cp)
        if matrix_type == "cupy_coo":
            data = sp.coo_matrix(data)
            dtype_cp = data_types_cp_mapping[dtype]
            return cu_sp.coo_matrix(data, dtype=dtype_cp)
        if matrix_type == "cupy":
            dtype_cp = data_types_cp_mapping[dtype]
            return cp.asarray(data, dtype=dtype_cp)

        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    return _make_matrix
