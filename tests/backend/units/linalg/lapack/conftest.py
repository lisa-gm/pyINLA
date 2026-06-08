# tests/backend/units/linalg/lapack/conftest.py

import pytest


from dalia.backend.config import cupy_version, nvmath_version

import numpy as np

if cupy_version is not None:
    import cupy as cp

# Type groups - reusable across all tests
INTERNAL_DEVICE_TYPES = ["host"]
DATA_TYPES = ["float32", "float64", "complex64", "complex128"]

INTERNAL_DEVICE_TYPES.append(pytest.param("accelerator", marks=pytest.mark.skipif(
                cupy_version is None,
                reason="CuPy is not installed",
            ),))

@pytest.fixture
def array_factory():
    
    def _make_array(data_type, shape=(3, 3), data=None, device_type="host"):
        if data is None:
            data = np.arange(1, shape[0] * shape[1] + 1).reshape(shape)

        if data_type == "float32":
            data_type = np.float32
        elif data_type == "float64":
            data_type = np.float64
        elif data_type == "complex64":
            data_type = np.complex64
        elif data_type == "complex128":
            data_type = np.complex128
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        data = data.astype(data_type)

        if device_type == "host":
            return data
        elif device_type == "accelerator":
            return cp.asarray(data, dtype=data_type)
        else:
            raise ValueError(f"Unknown device type: {device_type}")
        
    return _make_array