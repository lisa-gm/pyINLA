"""

Forked and modified from scipy.linalg.blas: https://github.com/scipy/scipy/blob/v1.15.3/scipy/linalg/_basic.py#L411

"""

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=protected-access

from typing import Literal

import numpy as np
from scipy.linalg.blas import get_blas_funcs

from dalia.backend.datastructures import Matrix, Vector
from dalia.backend.datastructures.matrix.core.dense import DenseMatrix


def gemv(
    trans_a: Literal["N", "n", "T", "t", "C", "c"],
    alpha: float,
    a: Matrix,
    x: Vector,
    beta: float,
    y: Vector = None,
    hw_target: Literal["default", "host", "accelerator"] = "default",
) -> Vector | None:
    """Wrapper for performing general matrix-vector products on
    Matrix/Vector datastructures.

    This routine performs one of the following general matrix-vector operations:
        y = alpha * op(A) @ x + beta * y
    where op(A) is one of
        op(A) = A        if trans_a == 'N'
        op(A) = A.T/H    if trans_a == 'T' or 'C'

    API: Wrapper for Matrix/Vector datastructures, arguments extended from the BLAS convention.

    Parameters
    ----------
    trans_a : {'N', 'n', 'T', 't', 'C', 'c'}
        Specifies the operation to be performed on matrix `a`. 'N' or 'n' for
        no transpose, 'T' or 't' or 'C' or 'c' for transpose/conjugate transpose.
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
    - Only supports single right-hand side (single column vector x).
      For multiple RHS, use GEMM instead.
    """
    # General assertion on input types
    if not isinstance(a, Matrix):
        raise TypeError(f"Invalid type for a, given: {type(a)}, expected: Matrix")
    if not isinstance(x, Vector):
        raise TypeError(f"Invalid type for x, given: {type(x)}, expected: Vector")
    if y is not None and not isinstance(y, Vector):
        raise TypeError(f"Invalid type for y, given: {type(y)}, expected: Vector")

    # . extra check as for now only support DenseMatrix
    if not isinstance(a, DenseMatrix):
        raise NotImplementedError(
            f"gemv currently only supports DenseMatrix, given: {type(a)}"
        )

    # Extract data arrays
    a_data = a._data
    x_data = x._data
    # . shape assertions
    if a_data.ndim != 2:
        raise ValueError(f"Matrix a must be 2D, given: {a_data.ndim}D")

    # . if x is 2D (multiple RHS), fall back to GEMM
    if x_data.ndim == 2:
        from dalia.backend.blas.l3.gemm import gemm as _gemm_fallback

        result = _gemm_fallback(
            trans_a=trans_a,
            trans_b="N",
            alpha=alpha,
            a=a,
            b=x,
            beta=beta,
            c=y,
            hw_target=hw_target,
        )
        # . gemm returns DenseMatrix when c=None, wrap as Vector
        if y is None:
            return Vector(data=result._data, hw_target=result.hw_target)
        return None

    # . adapt the shape assertions given the trans_a parameter
    if trans_a in ["N", "n"]:
        if a_data.shape[1] != x_data.shape[0]:
            raise ValueError(
                f"Shapes of a {a_data.shape} and x {x_data.shape} are incompatible "
                "for matrix-vector multiplication y = alpha * A @ x + beta * y"
            )
    else:
        if a_data.shape[0] != x_data.shape[0]:
            raise ValueError(
                f"Shapes of a {a_data.shape} and x {x_data.shape} are incompatible "
                "for matrix-vector multiplication y = alpha * A.T/H @ x + beta * y"
            )
    # . if y:Vector is not provided, allocate a new
    # data array (in F order) to store the result.
    y_data: np.ndarray = None
    if y is not None:
        y_data = y._data
        # . basic shape assertions
        row_dim = a_data.shape[0] if trans_a in ["N", "n"] else a_data.shape[1]
        if y_data.shape[0] != row_dim:
            raise ValueError(
                f"Shapes of a {a_data.shape}, x {x_data.shape}, and y {y_data.shape} "
                "are incompatible for the operation y = alpha * op(A) @ x + beta * y"
            )
    else:
        row_dim = a_data.shape[0] if trans_a in ["N", "n"] else a_data.shape[1]
        y_shape: tuple = (row_dim,) if x_data.ndim == 1 else (row_dim, x_data.shape[1])
        y_data = np.zeros(y_shape, dtype=a_data.dtype, order="F")

    # Sanitize hw_target
    # . this needs to be unified throughout the
    # backend and the BLAS part in particular
    if hw_target == "default":
        hw_target = a.hw_target

    if hw_target == "host":
        _gemv_host(
            trans_a=trans_a.upper(),
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
        return None

    elif hw_target == "accelerator":
        raise NotImplementedError(
            "Accelerator support for gemv is not implemented yet. Please use the host target."
        )
    else:
        raise ModuleNotFoundError("Unknown Module")


# Host-side Kernels
def _gemv_host(
    trans_a: Literal["N", "T", "C"],
    alpha: float,
    a: np.ndarray,
    x: np.ndarray,
    beta: float,
    y: np.ndarray,
) -> None:
    """Call the appropriate BLAS function for general matrix-vector product.

    API: Direct array interface, argument order matching LAPACK conventions.

    Parameters
    ----------
    trans_a : {'N', 'T', 'C'}
        Specifies the operation to be performed on matrix `a`. 'N' for no
        transpose, 'T' for transpose, 'C' for conjugate transpose.
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
    trans_a = {"N": 0, "T": 1, "C": 2}.get(trans_a, trans_a)

    _gemv = get_blas_funcs(("gemv"), (a, x))

    _gemv(
        trans=trans_a,
        alpha=alpha,
        a=a,
        x=x,
        beta=beta,
        y=y,
        overwrite_y=True,
    )
