""" """

# pylint: disable=protected-access

from typing import Literal

import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas.l3 import xxrk
from dalia.backend.datastructures.matrix.core.dense import DenseMatrix

from ..conftest import INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("uplo", ["U", "L"])
@pytest.mark.parametrize("trans_a", ["N", "T", "C"])
@pytest.mark.parametrize("alpha", [-1.5, 0.0, 1.5])
@pytest.mark.parametrize("beta", [-1.5, 0.0, 1.5])
def test_xxrk(
    matrix_factory: callable,
    data_type: Literal["float64", "complex128"],
    device_type: Literal["host", "accelerator"],
    inplace: bool,
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
    inplace: bool
        If True, provide an existing matrix `c` and verify it is modified in-place
        (the function returns None). If False, pass `c=None` and verify the function
        allocates and returns a new DenseMatrix.
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

    if inplace:
        c = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")
        beta_final = beta
    else:
        c = None
        beta_final = 0.0

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

    if inplace:
        c_reference_data = c._data.copy()
        c_reference_data[tri_indices(c._data)] *= beta
    else:
        c_reference_data = xp.zeros_like(a_reference_data)
        n = a_reference_data.shape[0] if trans_a == "N" else a_reference_data.shape[1]
        c_reference_data = xp.zeros((n, n), dtype=a_reference_data.dtype)

    if trans_a == "N":
        expected = (
            tri(alpha * a_reference_data @ a_reference_data.conj().T) + c_reference_data
        )
    else:
        expected = (
            tri(alpha * a_reference_data.conj().T @ a_reference_data) + c_reference_data
        )

    result = xxrk(
        uplo=uplo,
        trans_a=trans_a,
        alpha=alpha,
        a=a,
        beta=beta_final,
        c=c,
        hw_target=device_type,
    )

    if inplace:
        # . verify the function returned None for in-place mode
        assert result is None, f"Expected None for inplace=True, got {type(result)}"
        obtained_data = c._data
    else:
        # . verify the function returned a DenseMatrix for allocate mode
        assert result is not None, "Expected a DenseMatrix for inplace=False, got None"
        assert isinstance(
            result, DenseMatrix
        ), f"Expected DenseMatrix for inplace=False, got {type(result)}"
        obtained_data = result._data

    # Verify the result is correct
    if device_type == "accelerator":
        obtained_data = obtained_data.get()
        expected = expected.get()

    assert np.allclose(obtained_data, expected)
