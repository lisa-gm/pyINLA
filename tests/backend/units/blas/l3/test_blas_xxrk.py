import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas.l3 import xxrk

from ..conftest import DATA_TYPES, INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("data_type", DATA_TYPES)
@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
@pytest.mark.parametrize("uplo", ["U", "L"])
@pytest.mark.parametrize("trans_a", ["N", "T"])
@pytest.mark.parametrize("alpha", [-1.5, 0.0, 1.5])
@pytest.mark.parametrize("beta", [-1.5, 0.0, 1.5])
def test_xxrk(matrix_factory, device_type, data_type, uplo, trans_a, alpha, beta):
    """Test the symmetric/hermitian rank-k update (SYHERK) operation.

    Notes:
    - This is not testign the complex part (herk) at all
    - the hw_device is hard coded to "host"
    """
    if (
        nvmath_version is None
        and data_type in ["complex64", "complex128"]
        and device_type == "accelerator"
    ):
        pytest.skip("nvmath needed for HERK")

    # . make operands
    A = matrix_factory("DenseMatrix", hw_target="host")
    C = matrix_factory("DenseMatrix", hw_target="host")

    # . TODO: this need work, this is hard-coded for now
    # -> The hw_target is hard-set to "host", the binding doesn't work for Nvidia accelerator yet
    if device_type == "host":
        xp = np
    elif device_type == "accelerator":
        xp = cp

    # . adapt reference for uplo parameter
    if uplo == "U":
        tri = xp.triu
        tri_indices = xp.triu_indices_from
    else:
        tri = xp.tril
        tri_indices = xp.tril_indices_from

    a_reference_data = A._data.copy()
    c_reference_data = C._data.copy()
    c_reference_data[tri_indices(C._data)] *= beta

    if trans_a == "N":
        expected = (
            tri(alpha * a_reference_data @ a_reference_data.conj().T) + c_reference_data
        )
    else:
        expected = (
            tri(alpha * a_reference_data.conj().T @ a_reference_data) + c_reference_data
        )

    # . only test in-place for now
    xxrk(
        uplo=uplo,
        trans_a=trans_a,
        alpha=alpha,
        a=A,
        beta=beta,
        c=C,
        hw_target=device_type,
    )

    # Verify the result is correct
    if device_type == "accelerator":
        C = C.get()
        expected = expected.get()

    assert np.allclose(C._data, expected)
