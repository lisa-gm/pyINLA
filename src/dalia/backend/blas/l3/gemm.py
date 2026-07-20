"""

Forked and modified from cupy.cublas.gemm: https://github.com/cupy/cupy/blob/3a2c950d64ee707096bc7ca1bf0b953a08206384/cupy/cublas.py#L689
and scipy.linal.solve_triangular: https://github.com/scipy/scipy/blob/v1.15.3/scipy/linalg/_basic.py#L411

"""

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=protected-access

from typing import Literal

import numpy as np
from scipy.linalg.blas import get_blas_funcs

# from dalia.backend.config import cupy_version
from dalia.backend.datastructures import DenseMatrix, Matrix

# if cupy_version is not None:
#     import cupy as cp
#     from cupy import _core
#     from cupy.cuda import device
#     from cupy_backends.cuda.libs import cublas


def gemm(
    trans_a: Literal["N", "n", "T", "t", "C", "c"],
    trans_b: Literal["N", "n", "T", "t", "C", "c"],
    alpha: float,
    a: Matrix,
    b: Matrix,
    beta: float,
    c: Matrix = None,
    hw_target: Literal["default", "host", "accelerator"] = "default",
) -> Matrix | None:
    """Wrapper for performing general matrix-matrix multiplication (GEMM)
    on Matrix datastructures.

    This routine perrforms one of the matrix-matrix operations
        C = alpha * op(A) @ op(B) + beta * C
    where op(X) is one of
        op(X) = X   if trans_X == 'N'
        op(X) = X.T/H if trans_X == 'T' or 'C'

    Parameters
    ----------
    trans_a : {'N', 'n', 'T', 't', 'C', 'c'}
        Specifies the form of op(A) to be used in the matrix multiplication.
    trans_b : {'N', 'n', 'T', 't', 'C', 'c'}
        Specifies the form of op(B) to be used in the matrix multiplication.
    alpha : float
        Scalar multiplier for the product of op(A) and op(B).
    a : Matrix
        The first input matrix.
    b : Matrix
        The second input matrix.
    beta : float
        Scalar multiplier for the matrix C.
    c : Matrix, optional
        Output matrix that will be added to the result. If None, a new matrix will
        be created, otherwise the matrix c will be overwritten with the result.
    hw_target : {'default', 'host', 'accelerator'}, optional
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
    if not isinstance(b, Matrix):
        raise TypeError(f"Invalid type for b, given: {type(b)}, expected: Matrix")
    if c is not None and not isinstance(c, Matrix):
        raise TypeError(f"Invalid type for c, given: {type(c)}, expected: Matrix")

    # . extra check as for now only support DenseMatrix
    if not isinstance(a, DenseMatrix):
        raise NotImplementedError(
            f"gemm currently only supports DenseMatrix, given: {type(a)}"
        )
    if not isinstance(b, DenseMatrix):
        raise NotImplementedError(
            f"gemm currently only supports DenseMatrix, given: {type(b)}"
        )
    if c is not None and not isinstance(c, DenseMatrix):
        raise NotImplementedError(
            f"gemm currently only supports DenseMatrix for output, given: {type(c)}"
        )

    # Extract data arrays
    a_data = a._data
    b_data = b._data
    # . if c:Matrix is not provided, allocate a new
    # data array (in F order) to store the result.
    c_data: np.ndarray = None
    if c is not None:
        c_data = c._data
        # . basic shape assertions
        m = a_data.shape[0] if trans_a == "N" else a_data.shape[1]
        n = b_data.shape[1] if trans_b == "N" else b_data.shape[0]
        if c_data.shape[0] != m:
            raise ValueError(
                f"Shapes of a {a_data.shape}, b {b_data.shape}, and c {c_data.shape} "
                f"are incompatible for the operation C = alpha * op(A) @ op(B) + beta * C"
            )
        if c_data.shape[1] != n:
            raise ValueError(
                f"Shapes of a {a_data.shape}, b {b_data.shape}, and c {c_data.shape} "
                f"are incompatible for the operation C = alpha * op(A) @ op(B) + beta * C"
            )
    else:
        m = a_data.shape[0] if trans_a == "N" else a_data.shape[1]
        n = b_data.shape[1] if trans_b == "N" else b_data.shape[0]
        c_data = np.zeros((m, n), dtype=a_data.dtype, order="F")

    # Sanitize hw_target
    # . this needs to be unified throughout the
    # backend and the BLAS part in particular
    if hw_target == "default":
        hw_target = a.hw_target

    if hw_target == "host":
        _gemm_host(
            trans_a=trans_a.upper(),
            trans_b=trans_b.upper(),
            alpha=alpha,
            a=a_data,
            b=b_data,
            beta=beta,
            c=c_data,
        )

        # If c:Matrix wasn't provided, wrap and return
        # the result as a Matrix datastructure
        if c is None:
            return DenseMatrix(data=c_data, hw_target="host")
        return None

    elif hw_target == "accelerator":
        raise NotImplementedError(
            "Accelerator support for gemm is not implemented yet. Please use the host target."
        )
        # return matmul_gemm_accelerator(
        #     a=a_data,
        #     b=b_data,
        #     c=c_data,
        #     alpha=alpha,
        #     beta=beta,
        #     trans_a=trans_a,
        #     trans_b=trans_b,
        #     overwrite_c=overwrite_c,
        # )

    else:
        raise ModuleNotFoundError("Unknown Module")


# Host-side Kernels
def _gemm_host(
    trans_a: Literal["N", "T", "C"],
    trans_b: Literal["N", "T", "C"],
    alpha: float,
    a: np.ndarray,
    b: np.ndarray,
    beta: float,
    c: np.ndarray,
) -> None:
    """Call the appropriate BLAS function for general matrix-matrix multiplication.

    API: Direct array interface, argument order matching LAPACK conventions.

    Parameters
    ----------
    trans_a : {'N', 'T', 'C'}
        Specifies the operation to be performed on matrix `a`. 'N' for no
        transpose, 'T' for transpose, 'C' for conjugate transpose.
    trans_b : {'N', 'T', 'C'}
        Specifies the operation to be performed on matrix `b`. 'N' for no
        transpose, 'T' for transpose, 'C' for conjugate transpose.
    alpha : float
        Scalar multiplier for the product of op(A) and op(B).
    a : np.ndarray
        First input matrix.
    b : np.ndarray
        Second input matrix.
    beta : float
        Scalar multiplier for the matrix C.
    c : np.ndarray
        Output matrix that will be added to the result. This matrix will be modified in-place.

    Returns
    -------
    None
    - The result is stored in the `c` array, which is modified in-place.
    """
    trans_a = {"N": 0, "T": 1, "C": 2}.get(trans_a, trans_a)
    trans_b = {"N": 0, "T": 1, "C": 2}.get(trans_b, trans_b)
    (_gemm,) = get_blas_funcs(("gemm",), (a, b))

    _gemm(
        alpha=alpha,
        a=a,
        b=b,
        beta=beta,
        c=c,
        trans_a=trans_a,
        trans_b=trans_b,
        overwrite_c=True,
    )


# # Nvidia Accelerator-side Kernels
# # Util functions for cupy gemm
# def _trans_to_cublas_op(trans):
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


# def _change_order_if_necessary(a, lda):
#     if lda is None:
#         lda = a.shape[0]
#         if not a._f_contiguous:
#             a = a.copy(order="F")
#     return a, lda


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


# # Util functions for cupy gemm end


# # TODO: warnings for copies, maybe...
# def matmul_gemm_accelerator(
#     a, b, c=None, alpha=1.0, beta=0.0, transa=0, transb=0, overwrite_c=0
# ):
#     """Computes GEMM on a cuda accelerator"""
#     assert a.ndim == b.ndim == 2
#     assert a.dtype == b.dtype
#     dtype = a.dtype.char
#     if dtype == "f":
#         func = cublas.sgemm
#     elif dtype == "d":
#         func = cublas.dgemm
#     elif dtype == "F":
#         func = cublas.cgemm
#     elif dtype == "D":
#         func = cublas.zgemm
#     else:
#         raise TypeError("invalid dtype")

#     transa = _trans_to_cublas_op(transa)
#     transb = _trans_to_cublas_op(transb)
#     if transa == cublas.CUBLAS_OP_N:
#         m, k = a.shape
#     else:
#         k, m = a.shape
#     if transb == cublas.CUBLAS_OP_N:
#         n = b.shape[1]
#         assert b.shape[0] == k
#     else:
#         n = b.shape[0]
#         assert b.shape[1] == k

#     out = None
#     if c is None:
#         out = cp.empty((m, n), dtype=dtype, order="F")
#         beta = 0.0
#     else:
#         if overwrite_c:
#             out = c
#         else:
#             out = c.copy(order="F")
#         assert out.ndim == 2
#         assert out.shape == (m, n)
#         assert out.dtype == dtype

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

#     lda, transa = _decide_ld_and_trans(a, transa)
#     ldb, transb = _decide_ld_and_trans(b, transb)
#     if not (lda is None or ldb is None):
#         if out._f_contiguous:
#             try:
#                 func(
#                     handle,
#                     transa,
#                     transb,
#                     m,
#                     n,
#                     k,
#                     alpha_ptr,
#                     a.data.ptr,
#                     lda,
#                     b.data.ptr,
#                     ldb,
#                     beta_ptr,
#                     out.data.ptr,
#                     m,
#                 )
#             finally:
#                 cublas.setPointerMode(handle, orig_mode)
#             return out
#         elif out._c_contiguous:
#             # Computes out.T = alpha * b.T @ a.T + beta * out.T
#             try:
#                 func(
#                     handle,
#                     1 - transb,
#                     1 - transa,
#                     n,
#                     m,
#                     k,
#                     alpha_ptr,
#                     b.data.ptr,
#                     ldb,
#                     a.data.ptr,
#                     lda,
#                     beta_ptr,
#                     out.data.ptr,
#                     n,
#                 )
#             finally:
#                 cublas.setPointerMode(handle, orig_mode)
#             return out

#     a, lda = _change_order_if_necessary(a, lda)
#     b, ldb = _change_order_if_necessary(b, ldb)
#     c = out
#     if not out._f_contiguous:
#         c = out.copy(order="F")
#     try:
#         func(
#             handle,
#             transa,
#             transb,
#             m,
#             n,
#             k,
#             alpha_ptr,
#             a.data.ptr,
#             lda,
#             b.data.ptr,
#             ldb,
#             beta_ptr,
#             c.data.ptr,
#             m,
#         )
#     finally:
#         cublas.setPointerMode(handle, orig_mode)
#     if not out._f_contiguous:
#         _core.elementwise_copy(c, out)
#     return out
