""" """

# pylint: disable=protected-access

from typing import Literal

import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas.l2 import gemv
from dalia.backend.datastructures.matrix.core.vector_dense import Vector

from ..conftest import INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("trans_a", ["N", "T", "C"])
@pytest.mark.parametrize("alpha", [-1.5, 0.0, 1.5])
@pytest.mark.parametrize("beta", [-1.5, 0.0, 1.5])
@pytest.mark.parametrize("n_rhs", [1, 2])
def test_gemv(
    matrix_factory: callable,
    data_type: Literal["float64", "complex128"],
    device_type: Literal["host", "accelerator"],
    inplace: bool,
    trans_a: Literal["N", "T", "C"],
    alpha: float,
    beta: float,
    n_rhs: int,
):
    """Test the general matrix-vector product (GEMV) operation.

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
    trans_a: Literal["N", "T", "C"]
        Specifies whether to transpose or conjugate transpose the matrix A.
    alpha: float
        Scalar multiplier for the product of op(A) and x.
    beta: float
        Scalar multiplier for the existing vector y.

    Assert
    ------
    The result of the gemv operation is compared against a reference
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
        pytest.skip("nvmath needed for complex GEMV on accelerator")

    # . TODO: this need work, this is hard-coded for now
    # -> The hw_target is hard-set to "host", the binding doesn't work for Nvidia accelerator yet
    if device_type == "host":
        xp = np
    elif device_type == "accelerator":
        xp = cp

    # . make operands: (3, 5) matrix A and appropriately sized vector x
    a = matrix_factory("DenseMatrix", shape=(3, 5), dtype=data_type, hw_target="host")
    # . x length: a.shape[1] for trans_a="N", a.shape[0] for trans_a="T"/"C"
    x_len = a._data.shape[1] if trans_a == "N" else a._data.shape[0]
    x_data = xp.arange(1, x_len * n_rhs + 1, dtype=xp.float64).astype(a._data.dtype)
    x_data = x_data.reshape(x_len, n_rhs)
    if n_rhs == 1:
        x_data = x_data.ravel()
    x = Vector(data=x_data, hw_target="host")
    x_vec = xp.atleast_1d(x._data)

    if inplace:
        if trans_a in ["T", "C"]:
            y_len = a._data.shape[1]
        else:
            y_len = a._data.shape[0]
        y_data_arr = xp.arange(1, y_len * n_rhs + 1, dtype=xp.float64).astype(
            a._data.dtype
        )
        y_data_arr = y_data_arr.reshape(y_len, n_rhs)
        if n_rhs == 1:
            y_data_arr = y_data_arr.ravel()
        y = Vector(data=y_data_arr, hw_target="host")
        beta_final = beta
    else:
        y = None
        beta_final = 0.0

    a_reference_data = a._data.copy()
    x_reference_data = x_vec.copy()

    if inplace:
        y_reference_data = y._data.copy()
    else:
        y_len = (
            a_reference_data.shape[1]
            if trans_a in ["T", "C"]
            else a_reference_data.shape[0]
        )
        y_shape = (y_len, n_rhs) if n_rhs > 1 else (y_len,)
        y_reference_data = xp.zeros(y_shape, dtype=a_reference_data.dtype)

    # . apply op(A) according to trans parameter
    if trans_a in ["T", "C"]:
        a_op = a_reference_data.conj().T
    else:
        a_op = a_reference_data

    expected = alpha * a_op @ x_reference_data + beta_final * y_reference_data

    result = gemv(
        trans_a=trans_a,
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
        obtained_data = y._data
    else:
        # . verify the function returned a Vector for allocate mode
        assert result is not None, "Expected a Vector for inplace=False, got None"
        assert isinstance(
            result, Vector
        ), f"Expected Vector for inplace=False, got {type(result)}"
        obtained_data = result._data

    # Verify the result is correct
    if device_type == "accelerator":
        obtained_data = obtained_data.get()
        expected = expected.get()

    assert np.allclose(obtained_data.ravel(), expected.ravel())
