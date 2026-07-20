"""
Wrapper for performing symmetric (symv) and hermitian (hemv) matrix vector
products on Matrix/Vector datastructures.
"""

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=protected-access

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
    """Wrapper for performing symmetric (symv) and hermitian (hemv) matrix vector
    products on Matrix/Vector datastructures.

    This routine performs the following symmetric (hermitian) operation:
        y = alpha * A @ x + beta * y
    were A is a symmetric (hermitian) matrix, x and y are vectors, and alpha and beta are scalars.

    API: Wrapper for Matrix datastructures, arguments extended from the BLAS convention.

    Parameters
    ----------
    uplo : {'U', 'u', 'L', 'l'}
        Specifies whether the upper or lower triangular part of the result is
        to be referenced. 'U' or 'u' for upper, 'L' or 'l' for lower.
    alpha : float
        Scalar to be multiplied with matrix `a`.
    a : Matrix
        Matrix to apply to the vector `x`.
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

    Notes
    -----
    - Need to sanitize the number of RHS (xxmv only support single RHS)
    - If several RHS are given, fall-back to SYMM/HEMM (xxmv is a special case of SYMM/HEMM)
    """
    # General assertion on input types
    if not isinstance(a, Matrix):
        raise TypeError(f"Invalid type for a, given: {type(a)}, expected: Matrix")
    if x is not None and not isinstance(x, Vector):
        raise TypeError(f"Invalid type for x, given: {type(x)}, expected: Vector")
    if y is not None and not isinstance(y, Vector):
        raise TypeError(f"Invalid type for y, given: {type(y)}, expected: Vector")

    # . extra check as for now only support DenseMatrix
    if not isinstance(a, DenseMatrix):
        raise NotImplementedError(
            f"xxmv currently only supports DenseMatrix, given: {type(a)}"
        )

    # Extract data-arrays
    a_data: np.ndarray = a._data
    x_data: np.ndarray = x._data
    # . shape assertions
    if a_data.ndim != 2:
        raise ValueError(f"Matrix a must be 2D, given: {a_data.ndim}D")
    if a_data.shape[0] != a_data.shape[1]:
        raise ValueError(f"Matrix a must be square, given: {a_data.shape}")
    if a_data.shape[1] != x_data.shape[0]:
        raise ValueError(
            f"Shapes of a {a_data.shape} and x {x_data.shape} are incompatible for matrix-vector multiplication"
        )
    # . if y:Vector is not provided, allocate a new
    # data array (in F order) to store the result.
    y_data: np.ndarray = None
    if y is not None:
        y_data = y._data
        # . basic shape assertions
        if a_data.shape[0] != y_data.shape[0]:
            raise ValueError(f"Shapes of a {a_data.shape} and y {y_data.shape} are \
                incompatible for vector-vector addition in the operation \
                y = alpha * A @ x + beta * y")
        # . additional shape assertions for multi-dimensional y (multiple RHS)
        if x_data.ndim != y_data.ndim:
            raise ValueError(f"Shapes of x {x_data.shape} and y {y_data.shape} are \
                incompatible for vector-vector addition in the operation \
                y = alpha * A @ x + beta * y")
        if x_data.ndim == 2 and x_data.shape[1] != y_data.shape[1]:
            raise ValueError(
                f"Shapes of x {x_data.shape} and y {y_data.shape} are incompatible for vector-vector addition in the operation y = alpha * A @ x + beta * y"
            )
    else:
        y_shape: tuple = (
            (a_data.shape[0],)
            if x_data.ndim == 1
            else (a_data.shape[0], x_data.shape[1])
        )
        y_data = np.zeros(y_shape, dtype=a_data.dtype, order="F")

    # Sanitize hw_target
    # . this needs to be unified throughout the
    # backend and the BLAS part in particular
    if hw_target == "default":
        hw_target = a.hw_target

    if hw_target == "host":
        _xxmv_host(
            uplo=uplo.upper(),
            alpha=alpha,
            a=a_data,
            x=x_data,
            beta=beta,
            y=y_data,
        )

        # If y:Vector wasn't provided, wrap and return
        # the result as a Vector datastructure
        if y is None:
            return Vector(data=y_data, hw_target="host")

    elif hw_target == "accelerator":
        raise NotImplementedError(
            "Accelerator support for xxmv is not implemented yet. Please use the host target."
        )
    else:
        raise ModuleNotFoundError("Unknown Module")


# Host-side Kernels
def _xxmv_host(
    uplo: Literal["U", "L"],
    alpha: float,
    a: np.ndarray,
    x: np.ndarray,
    beta: float,
    y: np.ndarray,
) -> None:
    """Call the appropriate BLAS function for symmetric/hermitian
    matrix-vector product based on the data type of `a` and `x`.

    API: Direct array interface, argument order matching LAPACK conventions.

    Parameters
    ----------
    uplo : {'U', 'L'}
        Specifies whether the upper or lower triangular part of the result is
        to be referenced. 'U' for upper, 'L' for lower.
    alpha : float
        Scalar to be multiplied with matrix `a`.
    a : np.ndarray
        Matrix to apply to the vector `x`.
    x : np.ndarray
        Vector to be multiplied with matrix `a`.
    beta : float
        Scalar to be multiplied with vector `y`.
    y : np.ndarray
        Output vector that will be added to the result. This vector will be modified in-place.

    Returns
    -------
    None
    - The result is stored in the `y` array, which is modified in-place.
    """

    if np.iscomplexobj(a):
        xxmv = get_blas_funcs(("hemv"), (a, x))
    else:
        xxmv = get_blas_funcs(("symv"), (a, x))

    xxmv(
        alpha=alpha,
        a=a,
        x=x,
        beta=beta,
        y=y,
        lower=uplo in ["L"],
        overwrite_y=True,
    )
