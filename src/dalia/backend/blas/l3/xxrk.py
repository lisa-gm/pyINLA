"""


Credits:
- The host-side was forked and adapted from scipy.linal.solve_triangular: https://github.com/scipy/scipy/blob/v1.15.3/scipy/linalg/_basic.py#L411
- The Nvidia accelerator-side was forked and adapted from cupy.cublas.syrk: https://github.com/cupy/cupy/blob/3a2c950d64ee707096bc7ca1bf0b953a08206384/cupy/cublas.py#L930

"""

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=protected-access

from typing import Literal

import numpy as np
from scipy.linalg.blas import get_blas_funcs

# from dalia.backend.config import cupy_version, nvmath_version
from dalia.backend.datastructures import Matrix
from dalia.backend.datastructures.matrix.core.dense import DenseMatrix

# from .gemm import matmul_gemm_accelerator

# if cupy_version is not None:
#     import cupy as cp
#     from cupy import _core
#     from cupy.cuda import device
#     from cupy_backends.cuda.libs import cublas

# if nvmath_version is not None:
#     from nvmath.bindings import cublas as nvcublas


def xxrk(
    uplo: Literal["U", "u", "L", "l"],
    trans_a: Literal["N", "n", "T", "t", "C", "c"],
    alpha: float,
    a: Matrix,
    beta: float,
    c: Matrix = None,
    hw_target: Literal["default", "host", "accelerator"] = "default",
) -> Matrix | None:
    """Wrapper for performing symmetric (syrk) and hermitian (herk) rank-k updates on
    Matrix datastructures.

    This routine performs one of the following symmetric (hermitian) rank k operations:
        C = alpha * op(A) @ op(A)^T/H + beta * C
    or
        C = alpha * op(A)^T/H @ op(A) + beta * C

    API: Wrapper for Matrix datastructures, arguments extended from the BLAS convention.

    Parameters
    ----------
    uplo : {'U', 'u', 'L', 'l'}
        Specifies whether the upper or lower triangular part of the result is
        to be referenced. 'U' or 'u' for upper, 'L' or 'l' for lower.
    trans_a : {'N', 'n', 'T', 't', 'C', 'c'}
        Specifies the operation to be performed on matrix `a`. 'N' or 'n' for
        no transpose, 'T' or 't' or 'C' or 'c' for transpose/conjugate transpose.
    alpha : float
        Scalar to be multiplied with matrix `a`.
    a : Matrix
        Matrix to be rank-updated.
    beta : float
        Scalar to be multiplied with matrix `c`.
    c : Matrix, optional
        Output matrix that will be added to the result. If None, a new matrix will
        be created, otherwise the matrix c will be overwritten with the result.
    hw_target : {'default', 'host', 'accelerator'}, default='default'
        Hardware target, either "host" or "accelerator" depending on the current
        location of matrix `a`. If set to "default", the function will automatically
        determine the hardware target based on the location of matrix `a`.

    Returns
    -------
    Matrix or None
        Resulting matrix after the rank-k update if `c` is not provided. If `c`
        is provided, it will be overwritten in-place with the result, and the
        routine will return None.
    """
    # General assertion on input types
    if not isinstance(a, Matrix):
        raise TypeError(f"Invalid type for a, given: {type(a)}, expected: Matrix")
    if c is not None and not isinstance(c, Matrix):
        raise TypeError(f"Invalid type for c, given: {type(c)}, expected: Matrix")

    # . extra check as for now only support DenseMatrix
    if not isinstance(a, DenseMatrix):
        raise NotImplementedError(
            f"xxrk currently only supports DenseMatrix, given: {type(a)}"
        )
    if c is not None and not isinstance(c, DenseMatrix):
        raise NotImplementedError(
            f"xxrk currently only supports DenseMatrix for output, given: {type(c)}"
        )

    # Extract data arrays
    a_data = a._data
    # . shape assertions
    if a_data.ndim != 2:
        raise ValueError(f"Matrix a must be 2D, given: {a_data.ndim}D")
    # . if c:Matrix is not provided, allocate a new
    # data array (in F order) to store the result.
    c_data: np.ndarray = None
    if c is not None:
        c_data = c._data
        # . basic shape assertions
        if c_data.shape[0] != c_data.shape[1]:
            raise ValueError(f"Matrix c must be square, given: {c_data.shape}")
        if trans_a in ["N", "n"]:
            if c_data.shape[0] != a_data.shape[0]:
                raise ValueError(
                    f"Shapes of a {a_data.shape} and c {c_data.shape} are incompatible for the operation C = alpha * A @ A^T/H + beta * C"
                )
        else:
            if c_data.shape[0] != a_data.shape[1]:
                raise ValueError(
                    f"Shapes of a {a_data.shape} and c {c_data.shape} are incompatible for the operation C = alpha * A^T/H @ A + beta * C"
                )
    else:
        c_shape: tuple = (
            (a_data.shape[0], a_data.shape[0])
            if trans_a in ["N", "n"]
            else (a_data.shape[1], a_data.shape[1])
        )
        c_data = np.zeros(c_shape, dtype=a_data.dtype, order="F")

    # Sanitize hw_target
    # . this needs to be unified throughout the
    # backend and the BLAS part in particular
    if hw_target == "default":
        hw_target = a.hw_target

    if hw_target == "host":
        _xxrk_host(
            uplo=uplo.upper(),
            trans_a=trans_a.upper(),
            alpha=alpha,
            a=a_data,
            beta=beta,
            c=c_data,
        )

        # If c:Matrix wasn't provided, wrap and return
        # the result as a Matrix datastructure
        if c is None:
            return DenseMatrix(data=c_data, hw_target="host")

    elif hw_target == "accelerator":
        raise NotImplementedError(
            "Accelerator support for xxrk is not implemented yet. Please use the host target."
        )
        # return _xxrk_accelerator(
        #     a=a_data,
        #     c=c_data,
        #     alpha=alpha,
        #     beta=beta,
        #     trans=trans_a,
        #     lower=lower,
        #     overwrite_c=overwrite_c,
        # )
    else:
        raise ModuleNotFoundError("Unknown Module")


# Host-side Kernels
def _xxrk_host(
    uplo: Literal["U", "L"],
    trans_a: Literal["N", "T", "C"],
    alpha: float,
    a: np.ndarray,
    beta: float,
    c: np.ndarray,
):
    """Call the appropriate BLAS function for symmetric/hermitian
    rank-k update based on the data type of `a` and `c`.

    API: Direct array interface, argument order matching LAPACK conventions.

    Parameters
    ----------
    uplo : {'U', 'L'}
        Specifies whether the upper or lower triangular part of the result is
        to be referenced. 'U' for upper, 'L' for lower.
    trans_a : {'N', 'T', 'C'}
        Specifies the operation to be performed on matrix `a`. 'N' for no
        transpose, 'T' for transpose, 'C' for conjugate transpose.
    alpha : float
        Scalar to be multiplied with matrix `a`.
    a : np.ndarray
        Matrix to be rank-updated.
    beta : float
        Scalar to be multiplied with matrix `c`.
    c : np.ndarray
        Output matrix that will be added to the result. This matrix will be modified in-place.

    Returns
    -------
    None
    - The result is stored in the `c` array, which is modified in-place.
    """

    # Get the suited blas function and adapt parameters accordingly
    if np.iscomplexobj(a):
        _xxrk = get_blas_funcs(("herk"), (a,))
        # . unify trans="T" and trans="C" for herk
        trans_a = {"N": 0, "T": 2, "C": 2}.get(trans_a, trans_a)
    else:
        _xxrk = get_blas_funcs(("syrk"), (a,))
        # . unify trans_a="T" and trans_a="C" for syrk
        trans_a = {"N": 0, "T": 1, "C": 1}.get(trans_a, trans_a)

    _xxrk(
        alpha=alpha,
        a=a,
        beta=beta,
        c=c,
        trans=trans_a,
        lower=uplo in ["L"],
        overwrite_c=True,
    )


# # Nvidia Accelerator-side Kernels
# def _trans_to_cublas_op(trans):
#     """Util functions for cuda xxrk"""
#     if trans == "N" or trans == cublas.CUBLAS_OP_N:
#         trans = cublas.CUBLAS_OP_N
#     elif trans == "T" or trans == cublas.CUBLAS_OP_T:
#         trans = cublas.CUBLAS_OP_T
#     elif trans == "C" or trans == cublas.CUBLAS_OP_C:
#         trans = cublas.CUBLAS_OP_C
#     else:
#         raise TypeError("invalid trans (actual: {})".format(trans))
#     return trans


# def _decide_ld_and_trans(a, trans):
#     ld = None
#     if trans in (cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T):
#         if a._f_contiguous:
#             ld = a.shape[0]
#         elif a._c_contiguous:
#             ld = a.shape[1]
#             trans = 1 - trans
#     return ld, trans


# def _get_scalar_ptr(a, dtype):
#     if isinstance(a, cp.ndarray):
#         if a.dtype != dtype:
#             a = cp.array(a, dtype=dtype)
#         a_ptr = a.data.ptr
#     else:
#         if not (isinstance(a, np.ndarray) and a.dtype == dtype):
#             a = np.array(a, dtype=dtype)
#         a_ptr = a.ctypes.data
#     return a, a_ptr


# def _xxrk_accelerator(
#     a, c=None, alpha=1.0, beta=0.0, trans="N", lower=False, overwrite_c=0
# ):
#     """Computes SYRK and HERK on a cuda accelerator

#     if nvmath is not installed HERK will call GEMM instead
#     """
#     assert a.ndim == 2
#     dtype = a.dtype.char
#     if dtype == "f":
#         func = cublas.ssyrk
#     elif dtype == "d":
#         func = cublas.dsyrk
#     elif dtype == "F":
#         try:
#             func = cublas.cherk
#         except AttributeError:
#             if nvmath_version is not None:
#                 func = nvcublas.cherk
#             else:
#                 matmul_gemm_accelerator(a, a, out, trans_b="C", alpha=alpha, beta=beta)
#     elif dtype == "D":
#         try:
#             func = cublas.zherk
#         except AttributeError:
#             if nvmath_version is not None:
#                 func = nvcublas.zherk
#             else:
#                 matmul_gemm_accelerator(a, a, out, trans_b="C", alpha=alpha, beta=beta)
#     else:
#         raise TypeError("invalid dtype")

#     trans = _trans_to_cublas_op(trans)
#     if trans == cublas.CUBLAS_OP_N:
#         n, k = a.shape
#     else:
#         k, n = a.shape
#     out = None
#     if c is None:
#         out = cp.zeros((n, n), dtype=dtype, order="F")
#         beta = 0.0
#     else:
#         if overwrite_c:
#             out = c
#         else:
#             out = c.copy(order="F")
#         assert out.ndim == 2
#         assert out.shape == (n, n)
#         assert out.dtype == dtype

#     if lower:
#         uplo = cublas.CUBLAS_FILL_MODE_LOWER
#     else:
#         uplo = cublas.CUBLAS_FILL_MODE_UPPER

#     alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
#     beta, beta_ptr = _get_scalar_ptr(beta, a.dtype)
#     handle = device.get_cublas_handle()
#     orig_mode = cublas.getPointerMode(handle)
#     if isinstance(alpha, cp.ndarray) or isinstance(beta, cp.ndarray):
#         if not isinstance(alpha, cp.ndarray):
#             alpha = cp.array(alpha)
#             alpha_ptr = alpha.data.ptr
#         if not isinstance(beta, cp.ndarray):
#             beta = cp.array(beta)
#             beta_ptr = beta.data.ptr
#         cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
#     else:
#         cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

#     lda, trans = _decide_ld_and_trans(a, trans)
#     ldo, _ = _decide_ld_and_trans(out, trans)

#     if out._c_contiguous:
#         if not a._c_contiguous:
#             a = a.copy(order="C")
#             trans = 1 - trans
#             lda = a.shape[1]
#         try:
#             func(
#                 handle,
#                 1 - uplo,
#                 trans,
#                 n,
#                 k,
#                 alpha_ptr,
#                 a.data.ptr,
#                 lda,
#                 beta_ptr,
#                 out.data.ptr,
#                 ldo,
#             )
#         finally:
#             cublas.setPointerMode(handle, orig_mode)

#     else:
#         if not a._f_contiguous:
#             a = a.copy(order="F")
#             lda = a.shape[0]
#             trans = 1 - trans
#         c = out
#         if not out._f_contiguous:
#             c = out.copy(order="F")
#         try:
#             func(
#                 handle,
#                 uplo,
#                 trans,
#                 n,
#                 k,
#                 alpha_ptr,
#                 a.data.ptr,
#                 lda,
#                 beta_ptr,
#                 out.data.ptr,
#                 ldo,
#             )
#         finally:
#             cublas.setPointerMode(handle, orig_mode)
#         if not out._f_contiguous:
#             out[...] = c
#     return out
