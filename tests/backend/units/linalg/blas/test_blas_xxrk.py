import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas import xxrk

from .conftest import DATA_TYPES, INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("data_type", DATA_TYPES)
@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
def test_xxrk(matrix_factory, device_type, data_type):
    """Test the symmetric/hermitian rank-k update (SYHERK) operation."""
    if (
        nvmath_version is None
        and data_type in ["complex64", "complex128"]
        and device_type == "accelerator"
    ):
        pytest.skip("nvmath needed for HERK")

    # . make operands
    A = matrix_factory("DenseMatrix", hw_target="host")
    C = matrix_factory("DenseMatrix", hw_target="host")

    # . xxrk parameters
    alpha = 1.5
    beta = 1.5

    if device_type == "host":
        xp = np
    elif device_type == "accelerator":
        xp = cp

    a_reference_data = A._data.copy()
    c_reference_data = C._data.copy()
    c_reference_data[xp.triu_indices_from(C._data)] *= beta
    expected = (
        xp.triu(alpha * a_reference_data @ a_reference_data.conj().T) + c_reference_data
    )

    # . only test in-place for now
    xxrk(uplo="U", trans_a="N", alpha=alpha, a=A, beta=beta, c=C, hw_target=device_type)

    # Verify the result is correct
    if device_type == "accelerator":
        C = C.get()
        expected = expected.get()
    assert np.allclose(C._data, expected)
