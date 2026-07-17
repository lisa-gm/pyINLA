import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas.l3 import gemm

from .conftest import DATA_TYPES, INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("data_type", DATA_TYPES)
@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
def test_gemm(array_factory, device_type, data_type):
    """Test the general matrix-matrix multiplication (GEMM) operation."""
    A = array_factory(data_type, shape=(3, 3), device_type=device_type)
    B = array_factory(data_type, shape=(3, 3), device_type=device_type)
    C = array_factory(data_type, shape=(3, 3), device_type=device_type)
    alpha = 1.5
    beta = 1.5

    expected = alpha * A @ B + beta * C
    X = gemm(A, B, device_type, c=C, alpha=alpha, beta=beta)
    # Verify the result is correct
    if device_type == "accelerator":
        X = X.get()
        expected = expected.get()
    assert np.allclose(X, expected)
