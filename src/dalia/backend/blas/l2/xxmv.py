from typing import Literal

import numpy as np
from scipy.linalg.blas import get_blas_funcs

from dalia.backend.datastructures import Matrix, Vector
from dalia.backend.datastructures.matrix.core.dense import DenseMatrix


def xxmv(
    uplo: Literal["U", "u", "L", "l"],
    alpha: float,
    a: Matrix,
    x: Vector,
    beta: float,
    y: Vector = None,
    hw_target: Literal["default", "host", "accelerator"] = "default",
) -> Matrix | None:
    """Wrapper for performing symmetric (syrk) and hermitian (herk) matrix vector
    products on Matrix/Vector datastructures.

    This routine performs the following symmetric (hermitian) operation:
        y = alpha * A @ x + beta * y

    Parameters
    ----------
    uplo : {'U', 'u', 'L', 'l'}
        Specifies whether the upper or lower triangular part of the result is
        to be referenced. 'U' or 'u' for upper, 'L' or 'l' for lower.
    alpha : float
        Scalar to be multiplied with matrix `a`.
    a : Matrix
        Matrix to be rank-updated.
    x : Vector
        Vector to be multiplied with matrix `a`.
    beta : float
        Scalar to be multiplied with vector `y`.
    y : Vector, optional
        Output vector that will be added to the result. If None, a new vector will
        be created, otherwise the vector y will be overwritten with the result.
    hw_target : {'default', 'host', 'accelerator'}, default='default'
        Hardware target, either "host" or "accelerator" depending on the current
        location of matrix `a`. If set to "default", the function will automatically
        determine the hardware target based on the location of matrix `a`.

    Returns
    -------
    Vector or None
        Resulting vector after the matrix-vector product if `y` is not provided. If `y`
        is provided, it will be overwritten in-place with the result, and the
        routine will return None.
    """
    # . assert operands types are valid
    if not isinstance(a, Matrix):
        raise TypeError(f"Invalid type for a, given: {type(a)}, expected: Matrix")
    if x is not None and not isinstance(x, Vector):
        raise TypeError(f"Invalid type for x, given: {type(x)}, expected: Vector")
    if y is not None and not isinstance(y, Vector):
        raise TypeError(f"Invalid type for y, given: {type(y)}, expected: Vector")

    # . extra check as for now only support DenseMatrix
    if not isinstance(a, DenseMatrix):
        raise NotImplementedError(
            f"xxrk currently only supports DenseMatrix, given: {type(a)}"
        )

    # . if y is given, perform in-place operation in y
    if y is not None:
        overwrite_y = True
    else:
        overwrite_y = False

    # . map uplo to lower boolean
    lower = uplo in ["L", "l"]

    # . extract underlying data from Matrix datastructures
    a_data = a._data
    x_data = x._data
    y_data = y._data if y is not None else None

    # . sanitize hw_target (if default make it the same hw_target as a)
    if hw_target == "default":
        hw_target = a.hw_target

    if hw_target == "host":
        return _xxmv_host(
            a=a_data,
            x=x_data,
            y=y_data,
            alpha=alpha,
            beta=beta,
            lower=lower,
            overwrite_y=overwrite_y,
        )
    elif hw_target == "accelerator":
        raise NotImplementedError(
            "Accelerator support for xxrk is not implemented yet. Please use the host target."
        )
    else:
        raise ModuleNotFoundError("Unknown Module")


# Host-side Kernels
def _xxmv_host(
    a,
    x,
    y,
    alpha,
    beta,
    lower,
    overwrite_y,
):
    """Computes SYMV and HEMV on the host

    additional Argument check_finite that checks if a is finite
    """

    if np.iscomplexobj(a):
        xxmv = get_blas_funcs(("hemv"), (a, x))
    else:
        xxmv = get_blas_funcs(("symv"), (a, x))

    if y is None:
        y = np.zeros_like(x)
        overwrite_y = True
        xxmv(alpha, a, x, beta, y, lower, overwrite_y)
        return Vector(data=y)

    return xxmv(alpha, a, x, beta, y, lower, overwrite_y)
