""" """

# pylint: disable=protected-access

from typing import Literal

import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas.l3 import gemm
from dalia.backend.datastructures.matrix.core.dense import DenseMatrix

from ..conftest import INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("trans_a", ["N", "T", "C"])
@pytest.mark.parametrize("trans_b", ["N", "T", "C"])
@pytest.mark.parametrize("alpha", [-1.5, 0.0, 1.5])
@pytest.mark.parametrize("beta", [-1.5, 0.0, 1.5])
def test_gemm(
    matrix_factory: callable,
    data_type: Literal["float64", "complex128"],
    device_type: Literal["host", "accelerator"],
    inplace: bool,
    trans_a: Literal["N", "T", "C"],
    trans_b: Literal["N", "T", "C"],
    alpha: float,
    beta: float,
):
    """Test the general matrix-matrix multiplication (GEMM) operation.

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
    trans_a: Literal["N", "T", "C"]
        Specifies whether to transpose or conjugate transpose matrix A.
    trans_b: Literal["N", "T", "C"]
        Specifies whether to transpose or conjugate transpose matrix B.
    alpha: float
        Scalar multiplier for the product of op(A) and op(B).
    beta: float
        Scalar multiplier for the existing matrix C.

    Assert
    ------
    The result of the gemm operation is compared against a reference
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
        pytest.skip("nvmath needed for complex GEMM on accelerator")

    # . make operands
    A = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")
    B = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")

    if inplace:
        C = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")
        beta_final = beta
    else:
        C = None
        beta_final = 0.0

    # . TODO: this need work, this is hard-coded for now
    # -> The hw_target is hard-set to "host", the binding doesn't work for Nvidia accelerator yet
    if device_type == "host":
        xp = np
    elif device_type == "accelerator":
        xp = cp

    a_reference_data = A._data.copy()
    b_reference_data = B._data.copy()

    if inplace:
        c_reference_data = C._data.copy()
    else:
        m = a_reference_data.shape[0] if trans_a == "N" else a_reference_data.shape[1]
        n = b_reference_data.shape[1] if trans_b == "N" else b_reference_data.shape[0]
        c_reference_data = xp.zeros((m, n), dtype=a_reference_data.dtype)

    # . apply op(A) and op(B) according to trans parameters
    if trans_a in ["T", "C"]:
        a_op = a_reference_data.conj().T
    else:
        a_op = a_reference_data

    if trans_b in ["T", "C"]:
        b_op = b_reference_data.conj().T
    else:
        b_op = b_reference_data

    expected = alpha * a_op @ b_op + beta_final * c_reference_data

    result = gemm(
        trans_a=trans_a,
        trans_b=trans_b,
        alpha=alpha,
        a=A,
        b=B,
        beta=beta_final,
        c=C,
        hw_target=device_type,
    )

    if inplace:
        # . verify the function returned None for in-place mode
        assert result is None, f"Expected None for inplace=True, got {type(result)}"
        obtained_data = C._data
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
