""" """

# pylint: disable=protected-access

from typing import Literal

import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas.l3 import xxrk

from ..conftest import INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
@pytest.mark.parametrize("uplo", ["U", "L"])
@pytest.mark.parametrize("trans_a", ["N", "T", "C"])
@pytest.mark.parametrize("alpha", [-1.5, 0.0, 1.5])
@pytest.mark.parametrize("beta", [-1.5, 0.0, 1.5])
def test_xxrk(
    matrix_factory: callable,
    data_type: Literal["float64", "complex128"],
    device_type: Literal["host", "accelerator"],
    uplo: Literal["U", "L"],
    trans_a: Literal["N", "T", "C"],
    alpha: float,
    beta: float,
):
    """Test the symmetric/hermitian rank-k update (SY/HERK) operation.

    Parameters
    ----------
    matrix_factory: callable
        A factory function to create matrices of different types.
    data_type: Literal["float64", "complex128"]
        The data type of the matrices.
    device_type: Literal["host", "accelerator"]
        The device type to run the test on.
    uplo: Literal["U", "L"]
        Specifies whether the upper or lower triangular part of the matrix is used.
    trans_a: Literal["N", "T", "C"]
        Specifies whether to transpose or conjugate transpose the matrix A.
    alpha: float
        Scalar multiplier for the rank-k update.
    beta: float
        Scalar multiplier for the existing matrix C.

    Assert
    ------
    The result of the xxrk operation is compared against a reference
    implementation using NumPy or CuPy on the raw data arrays.

    Notes
    -----
    - Do not support "accelerator" testing.
    """
    if (
        nvmath_version is None
        and data_type in ["complex64", "complex128"]
        and device_type == "accelerator"
    ):
        pytest.skip("nvmath needed for HERK")

    # . make operands
    a = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")
    c = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")

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

    a_reference_data = a._data.copy()
    c_reference_data = c._data.copy()
    c_reference_data[tri_indices(c._data)] *= beta

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
        a=a,
        beta=beta,
        c=c,
        hw_target=device_type,
    )

    # Verify the result is correct
    if device_type == "accelerator":
        c = c.get()
        expected = expected.get()

    assert np.allclose(c._data, expected)
