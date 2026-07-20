""" """

# pylint: disable=protected-access

from typing import Literal

import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas.l3 import trsm

from ..conftest import INTERNAL_DEVICE_TYPES


@pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("uplo", ["U", "L"])
@pytest.mark.parametrize("trans_a", ["N", "T", "C"])
@pytest.mark.parametrize("diag", ["U", "N"])
@pytest.mark.parametrize("alpha", [-1.5, 0.0, 1.5])
def test_trsm(
    matrix_factory: callable,
    data_type: Literal["float64", "complex128"],
    device_type: Literal["host", "accelerator"],
    side: Literal["L", "R"],
    uplo: Literal["U", "L"],
    trans_a: Literal["N", "T", "C"],
    diag: Literal["U", "N"],
    alpha: float,
):
    """Test the triangular solve matrix-matrix (TRSM) operation.

    Parameters
    ----------
    matrix_factory: callable
        A factory function to create matrices of different types.
    data_type: Literal["float64", "complex128"]
        The data type of the matrices.
    device_type: Literal["host", "accelerator"]
        The device type to run the test on.
    side: Literal["L", "R"]
        Specifies whether the triangular matrix `a` appears on the left or right side.
    uplo: Literal["U", "L"]
        Specifies whether the upper or lower triangular part of the matrix is used.
    trans_a: Literal["N", "T", "C"]
        Specifies whether to transpose or conjugate transpose matrix A.
    diag: Literal["U", "N"]
        Specifies whether the matrix `a` is unit triangular or not.
    alpha: float
        Scalar multiplier for the solution.

    Assert
    ------
    The result of the trsm operation is compared against a reference
    implementation using NumPy/CuPy's solve_triangular function.

    Notes
    -----
    - Do not support "accelerator" testing.
    - trsm is always in-place (modifies b directly).
    """
    if (
        nvmath_version is None
        and data_type in ["complex64", "complex128"]
        and device_type == "accelerator"
    ):
        pytest.skip("nvmath needed for complex TRSM on accelerator")

    # . TODO: this need work, this is hard-coded for now
    # -> The hw_target is hard-set to "host", the binding doesn't work for Nvidia accelerator yet
    if device_type == "host":
        xp = np
    elif device_type == "accelerator":
        xp = cp

    # . make operands: square triangular matrix A and RHS matrix B
    a = matrix_factory("DenseMatrix", dtype=data_type, hw_target="host")

    # . make A triangular: zero out the part not referenced
    if uplo == "U":
        tri_mask = xp.triu(xp.ones_like(a._data, dtype=bool), k=0)
    else:
        tri_mask = xp.tril(xp.ones_like(a._data, dtype=bool), k=0)
    a._data[~tri_mask] = 0.0

    # . for diag="U" (unit triangular), set diagonal to 1.0
    if diag == "U":
        xp.fill_diagonal(a._data, 1.0)

    # . make B: (n, 2) for side="L" or (2, n) for side="R"
    n = a._data.shape[0]
    b_shape = (n, 2) if side == "L" else (2, n)
    b_data = (
        xp.arange(1, n * 2 + 1, dtype=xp.float64).astype(a._data.dtype).reshape(b_shape)
    )
    b = matrix_factory("DenseMatrix", data=b_data, dtype=data_type, hw_target="host")

    # . reference: solve op(A) @ X = B using full dense linear solve
    a_reference_data = a._data.copy()
    if trans_a in ["T", "C"]:
        a_op = a_reference_data.conj().T
    else:
        a_op = a_reference_data

    # . compute reference: X = alpha * op(A)^{-1} @ B (or @ on the right)
    b_reference_data_original = b._data.copy()
    if side == "L":
        expected = alpha * xp.linalg.solve(a_op, b_reference_data_original)
    else:
        # side == "R": B @ op(A)^{-1} => solve(A^T, B^T)^T
        expected = alpha * xp.linalg.solve(a_op.T, b_reference_data_original.T).T

    # . call trsm (always in-place on b)
    trsm(
        side=side,
        uplo=uplo,
        trans_a=trans_a,
        diag=diag,
        alpha=alpha,
        a=a,
        b=b,
        hw_target=device_type,
    )

    # Verify the result is correct
    if device_type == "accelerator":
        b = b.get()
        expected = expected.get()

    assert np.allclose(b._data, expected)
