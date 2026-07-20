""" """

# pylint: disable=protected-access

from typing import Literal

import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas.l2 import xxmv
from dalia.backend.datastructures.matrix.core.vector_dense import Vector

from ..conftest import INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("uplo", ["U", "L"])
@pytest.mark.parametrize("alpha", [-1.5, 0.0, 1.5])
@pytest.mark.parametrize("beta", [-1.5, 0.0, 1.5])
def test_xxmv(
    matrix_factory: callable,
    data_type: Literal["float64", "complex128"],
    device_type: Literal["host", "accelerator"],
    inplace: bool,
    uplo: Literal["U", "L"],
    alpha: float,
    beta: float,
):
    """Test the symmetric/hermitian matrix-vector product (SYMV/HEMV) operation.

    Parameters
    ----------
    matrix_factory: callable
        A factory function to create matrices of different types.
    data_type: Literal["float64", "complex128"]
        The data type of the matrices.
    device_type: Literal["host", "accelerator"]
        The device type to run the test on.
    inplace: bool
        If True, provide an existing Vector `y` and verify it is modified in-place
        (the function returns None). If False, pass `y=None` and verify the function
        allocates and returns a new Vector.
    uplo: Literal["U", "L"]
        Specifies whether the upper or lower triangular part of the matrix is used.
    alpha: float
        Scalar multiplier for the matrix-vector product.
    beta: float
        Scalar multiplier for the existing vector y.

    Assert
    ------
    The result of the xxmv operation is compared against a reference
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
        pytest.skip("nvmath needed for HEMV")

    # . TODO: this need work, this is hard-coded for now
    # -> The hw_target is hard-set to "host", the binding doesn't work for Nvidia accelerator yet
    if device_type == "host":
        xp = np
    elif device_type == "accelerator":
        xp = cp

    # . make operands: (3, 3) square matrix A and (3,) vector x
    a = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")
    x_data = xp.arange(1, a._data.shape[0] + 1, dtype=xp.float64).astype(a._data.dtype)
    x = Vector(data=x_data.flatten(), hw_target="host")
    x_vec = xp.atleast_1d(x._data)

    # . make matrix symmetric/hermitian so that symv/hemv reads both triangles consistently
    if uplo == "U":
        tri_mask = xp.triu(xp.ones_like(a._data, dtype=bool), k=0)
    else:
        tri_mask = xp.tril(xp.ones_like(a._data, dtype=bool), k=0)
    a_sym = xp.where(tri_mask, a._data, a._data.conj().T)
    a._data[:] = a_sym

    # . prepare output vector y
    if inplace:
        y_len = a_sym.shape[0]
        y_data_arr = xp.arange(1, y_len + 1, dtype=xp.float64).astype(a_sym.dtype)
        y = Vector(data=y_data_arr.flatten(), hw_target="host")
        beta_final = beta
    else:
        y = None
        beta_final = 0.0

    a_reference_data = a_sym.copy()
    x_reference_data = x_vec.copy()

    if inplace:
        y_reference_data = y._data.copy().ravel()
    else:
        y_reference_data = xp.zeros(
            a_reference_data.shape[0], dtype=a_reference_data.dtype
        )

    expected = (
        alpha * a_reference_data @ x_reference_data + beta_final * y_reference_data
    )

    result = xxmv(
        uplo=uplo,
        alpha=alpha,
        a=a,
        x=Vector(data=x_reference_data, hw_target="host"),
        beta=beta_final,
        y=y,
        hw_target=device_type,
    )

    if inplace:
        # . verify the function returned None for in-place mode
        assert result is None, f"Expected None for inplace=True, got {type(result)}"
        obtained_data = y._data.ravel()
    else:
        # . verify the function returned a Vector for allocate mode
        assert result is not None, "Expected a Vector for inplace=False, got None"
        assert isinstance(
            result, Vector
        ), f"Expected Vector for inplace=False, got {type(result)}"
        obtained_data = result._data.ravel()

    # Verify the result is correct
    if device_type == "accelerator":
        obtained_data = obtained_data.get()
        expected = expected.get()

    assert np.allclose(obtained_data.ravel(), expected.ravel())
