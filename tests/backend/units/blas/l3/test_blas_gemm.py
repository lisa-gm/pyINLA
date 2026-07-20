import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas.l3 import gemm

from ..conftest import INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
@pytest.mark.parametrize("trans_a", ["N", "T"])
@pytest.mark.parametrize("trans_b", ["N", "T"])
@pytest.mark.parametrize("alpha", [-1.5, 0.0, 1.5])
@pytest.mark.parametrize("beta", [-1.5, 0.0, 1.5])
def test_gemm(
    matrix_factory,
    data_type,
    device_type,
    trans_a,
    trans_b,
    alpha,
    beta,
):
    """Test the general matrix-matrix multiplication (GEMM) operation."""
    # . make operands
    A = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")
    B = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")
    C = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")

    # Extract data array for reference computation
    a_reference_data = A._data.copy()
    b_reference_data = B._data.copy()
    c_reference_data = C._data.copy()

    # Compute expected result accordingly to trans parameters
    if trans_a == "T" and trans_b == "T":
        expected = (
            alpha * a_reference_data.conj().T @ b_reference_data.conj().T
            + beta * c_reference_data
        )
    elif trans_a == "T" and trans_b == "N":
        expected = (
            alpha * a_reference_data.conj().T @ b_reference_data
            + beta * c_reference_data
        )
    elif trans_a == "N" and trans_b == "T":
        expected = (
            alpha * a_reference_data @ b_reference_data.conj().T
            + beta * c_reference_data
        )
    else:
        expected = alpha * a_reference_data @ b_reference_data + beta * c_reference_data

    gemm(
        trans_a=trans_a,
        trans_b=trans_b,
        alpha=alpha,
        a=A,
        b=B,
        beta=beta,
        c=C,
        hw_target=device_type,
    )

    # Verify the result is correct
    if device_type == "accelerator":
        C = C.get()
        expected = expected.get()

    assert np.allclose(C._data, expected)
