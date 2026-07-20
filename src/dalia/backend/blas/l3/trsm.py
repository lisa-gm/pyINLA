""" """

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=protected-access

from typing import Literal

import numpy as np
from scipy.linalg.blas import get_blas_funcs

from dalia.backend.datastructures import DenseMatrix, Matrix


def trsm(
    side: Literal["L", "l", "R", "r"],
    uplo: Literal["U", "u", "L", "l"],
    trans_a: Literal["N", "n", "T", "t", "C", "c"],
    diag: Literal["U", "u", "N", "n"],
    alpha: float,
    a: Matrix,
    b: Matrix,
    hw_target: Literal["default", "host", "accelerator"] = "default",
) -> None:
    """Wrapper for performing triangular solve matrix-matrix on
    Matrix datastructures.

    This routine performs one of the following triangular solve matrix-matrix operations:
        B = alpha * op(A)^{-1} @ B  if side == 'L'
        B = alpha * B @ op(A)^{-1}  if side == 'R'
    where op(A) is one of
        op(A) = A        if trans_a == 'N'
        op(A) = A.T/H    if trans_a == 'T' or 'C'

    API: Wrapper for Matrix datastructures, arguments extended from the BLAS convention.

    Parameters
    ----------
    side : {'L', 'l', 'R', 'r'}
        Specifies whether the triangular matrix `a` appears on the left or right
        side of the operation. 'L' or 'l' for left, 'R' or 'r' for right.
    uplo : {'U', 'u', 'L', 'l'}
        Specifies whether the upper or lower triangular part of the matrix `a`
        is to be referenced. 'U' or 'u' for upper, 'L' or 'l' for lower.
    trans_a : {'N', 'n', 'T', 't', 'C', 'c'}
        Specifies the operation to be performed on matrix `a`. 'N' or 'n' for
        no transpose, 'T' or 't' or 'C' or 'c' for transpose/conjugate transpose.
    diag : {'U', 'u', 'N', 'n'}
        Specifies whether the matrix `a` is unit triangular or not. 'U' or 'u' for
        unit triangular, 'N' or 'n' for non-unit triangular.
    alpha : float
        Scalar to be multiplied with the solution.
    a : Matrix
        Triangular matrix to be used in the operation.
    b : Matrix
        Right-hand side matrix. The operation is performed in-place on this matrix.
    hw_target : {'default', 'host', 'accelerator'}, default='default'
        Specifies the hardware target for the operation.

    Returns
    -------
    None
        The operation is performed in-place on the matrix `b`.
    """
    # General assertion on input types
    if not isinstance(a, Matrix):
        raise TypeError(f"Invalid type for a, given: {type(a)}, expected: Matrix")
    if not isinstance(b, Matrix):
        raise TypeError(f"Invalid type for b, given: {type(b)}, expected: Matrix")

    # . extra check as for now only support DenseMatrix
    if not isinstance(a, DenseMatrix):
        raise NotImplementedError(
            f"trsm currently only supports DenseMatrix, given: {type(a)}"
        )
    if not isinstance(b, DenseMatrix):
        raise NotImplementedError(
            f"trsm currently only supports DenseMatrix, given: {type(b)}"
        )

    # Extract data arrays
    a_data = a._data
    b_data = b._data
    # . shape assertions
    if a_data.ndim != 2:
        raise ValueError(f"Matrix a must be 2D, given: {a_data.ndim}D")
    if a_data.shape[0] != a_data.shape[1]:
        raise ValueError(f"Triangular matrix a must be square, given: {a_data.shape}")
    if side in ["L", "l"]:
        if a_data.shape[1] != b_data.shape[0]:
            raise ValueError(
                f"Shapes of a {a_data.shape} and b {b_data.shape} are incompatible "
                "for the operation B = alpha * op(A)^{-1} @ B (side='L')"
            )
    else:
        if a_data.shape[0] != b_data.shape[1]:
            raise ValueError(
                f"Shapes of a {a_data.shape} and b {b_data.shape} are incompatible "
                "for the operation B = alpha * B @ op(A)^{-1} (side='R')"
            )

    # Sanitize hw_target
    # . this needs to be unified throughout the
    # backend and the BLAS part in particular
    if hw_target == "default":
        hw_target = a.hw_target

    if hw_target == "host":
        _trsm_host(
            side=side.upper(),
            uplo=uplo.upper(),
            trans_a=trans_a.upper(),
            diag=diag.upper(),
            alpha=alpha,
            a=a_data,
            b=b_data,
        )

        return None

    elif hw_target == "accelerator":
        raise NotImplementedError(
            "Accelerator support for trsm is not implemented yet. Please use the host target."
        )

    else:
        raise ModuleNotFoundError("Unknown Module")


# Host-side Kernels
def _trsm_host(
    side: Literal["L", "R"],
    uplo: Literal["U", "L"],
    trans_a: Literal["N", "T", "C"],
    diag: Literal["U", "N"],
    alpha: float,
    a: np.ndarray,
    b: np.ndarray,
) -> None:
    """Call the appropriate BLAS function for triangular solve matrix-matrix
    based on the data type of `a` and `b`.

    API: Direct array interface, argument order matching LAPACK conventions.

    Parameters
    ----------
    side : {'L', 'R'}
        Specifies whether the triangular matrix `a` appears on the left or right
        side of the operation.
    uplo : {'U', 'L'}
        Specifies whether the upper or lower triangular part of the matrix `a`
        is to be referenced.
    trans_a : {'N', 'T', 'C'}
        Specifies the operation to be performed on matrix `a`. 'N' for no
        transpose, 'T' for transpose, 'C' for conjugate transpose.
    diag : {'U', 'N'}
        Specifies whether the matrix `a` is unit triangular or not.
    alpha : float
        Scalar to be multiplied with the solution.
    a : np.ndarray
        Triangular matrix.
    b : np.ndarray
        Right-hand side matrix. This matrix will be modified in-place.

    Returns
    -------
    None
    - The result is stored in the `b` array, which is modified in-place.
    """
    trans_a = {"N": 0, "T": 1, "C": 2}.get(trans_a, trans_a)
    side = {"L": 0, "R": 1}.get(side, side)

    (_trsm,) = get_blas_funcs(("trsm",), (a, b))

    _trsm(
        alpha=alpha,
        a=a,
        b=b,
        side=side,
        lower=uplo in ["L"],
        trans_a=trans_a,
        diag=diag in ["U"],
        overwrite_b=True,
    )
