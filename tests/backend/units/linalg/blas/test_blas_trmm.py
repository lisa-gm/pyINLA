import pytest
import numpy as np
from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas import trmm

from .conftest import INTERNAL_DEVICE_TYPES, DATA_TYPES

@pytest.mark.parametrize("data_type", DATA_TYPES)
@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
def test_trmm(array_factory, device_type, data_type):
    """Test the triangular matrix-matrix multiplication (TRMM) operation."""
    if nvmath_version is None and device_type == "accelerator":
        pytest.skip("nvmath needed for TRMM")
    A = array_factory(data_type, shape=(3, 3), device_type=device_type)
    B = array_factory(data_type, shape=(3, 3), device_type=device_type)
    if device_type == "host":
        xp = np
    elif device_type == "accelerator":
        xp = cp
    alpha = 1.0    
    expected = alpha * xp.triu(A) @ B
    X = trmm(A, B, device_type, alpha=alpha)
    # Verify the result is correct
    if device_type == "accelerator":
        X = X.get()
        expected = expected.get()
    assert np.allclose(X, expected)