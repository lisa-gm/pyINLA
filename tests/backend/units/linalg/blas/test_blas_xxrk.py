import pytest
import numpy as np
from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas import gemm, xxrk, trmm

from .conftest import INTERNAL_DEVICE_TYPES, DATA_TYPES

@pytest.mark.parametrize("data_type", DATA_TYPES)
@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
def test_xxrk(array_factory, device_type, data_type):
    """Test the symmetric/hermitian rank-k update (SYHERK) operation."""
    if (
        nvmath_version is None
        and data_type in ["complex64", "complex128"]
        and device_type == "accelerator"
    ):
        pytest.skip("nvmath needed for HERK")
    A = array_factory(data_type, shape=(3, 3), device_type=device_type)
    C = array_factory(data_type, shape=(3, 3), device_type=device_type)
    alpha = 1.5
    beta = 1.5
    if device_type == "host":
        xp = np
    elif device_type == "accelerator":
        xp = cp
    C_copy = C.copy()
    C_copy[xp.triu_indices_from(C)] *= beta
    expected = xp.triu(alpha * A @ A.conj().T) + C_copy
    X = xxrk(A, device_type, c=C, alpha=alpha, beta=beta)
    # Verify the result is correct
    if device_type == "accelerator":
        X = X.get()
        expected = expected.get()
    assert np.allclose(X, expected)