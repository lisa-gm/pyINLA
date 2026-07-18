# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Forked and modified from cupy.cublas.gemm: https://github.com/cupy/cupy/blob/3a2c950d64ee707096bc7ca1bf0b953a08206384/cupy/cublas.py#L689
# and scipy.linal.solve_triangular: https://github.com/scipy/scipy/blob/v1.15.3/scipy/linalg/_basic.py#L411

from typing import Literal

import numpy as np
from scipy.linalg._decomp import _asarray_validated
from scipy.linalg._misc import _datacopied
from scipy.linalg.blas import get_blas_funcs

# from dalia.backend.config import cupy_version
from dalia.backend.datastructures import DenseMatrix, Matrix

# if cupy_version is not None:
#     import cupy as cp
#     from cupy import _core
#     from cupy.cuda import device
#     from cupy_backends.cuda.libs import cublas


def gemm(
    trans_a: Literal["N", "T", "C"],
    trans_b: Literal["N", "T", "C"],
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
    trans_a : {'N', 'T', 'C'}
        Specifies the form of op(A) to be used in the matrix multiplication.
    trans_b : {'N', 'T', 'C'}
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
    # . assert operands types are valid
    if not isinstance(a, Matrix):
        raise TypeError(f"Invalid type for a, given: {type(a)}, expected: Matrix")
    if not isinstance(b, Matrix):
        raise TypeError(f"Invalid type for b, given: {type(b)}, expected: Matrix")
    if c is not None and not isinstance(c, Matrix):
        raise TypeError(f"Invalid type for c, given: {type(c)}, expected: Matrix")

    # . extra check as for now only support DenseMatrix
    if not isinstance(a, DenseMatrix):
        raise NotImplementedError(
            f"xxrk currently only supports DenseMatrix, given: {type(a)}"
        )
    if not isinstance(b, DenseMatrix):
        raise NotImplementedError(
            f"xxrk currently only supports DenseMatrix, given: {type(b)}"
        )
    if c is not None and not isinstance(c, DenseMatrix):
        raise NotImplementedError(
            f"xxrk currently only supports DenseMatrix for output, given: {type(c)}"
        )

    # . if c is given, perform in-place operation in c
    if c is not None:
        overwrite_c = True
    else:
        overwrite_c = False

    # . extract underlying data from Matrix datastructures
    a_data = a._data
    b_data = b._data
    c_data = c._data if c is not None else None

    # Sanitize hw_target
    # . this needs to be unified throughout the
    # backend and the BLAS part in particular
    if hw_target == "default":
        hw_target = a.hw_target

    # . sanitize trans_a (make it uppercase)
    trans_a = trans_a.upper()
    trans_b = trans_b.upper()

    if hw_target == "host":
        return _gemm_host(
            a=a_data,
            b=b_data,
            c=c_data,
            alpha=alpha,
            beta=beta,
            trans_a=trans_a,
            trans_b=trans_b,
            overwrite_c=overwrite_c,
        )
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
        ModuleNotFoundError("Unknown Module")


# Host-side Kernels
def _gemm_host(
    a,
    b,
    c,
    alpha,
    beta,
    trans_a,
    trans_b,
    overwrite_c,
):
    """Computes GEMM on the host

    additional Argument check_finite that checks if a and b are finite
    """

    trans_a = {"N": 0, "T": 1, "C": 2}.get(trans_a, trans_a)
    trans_b = {"N": 0, "T": 1, "C": 2}.get(trans_b, trans_b)
    (gemm,) = get_blas_funcs(("gemm",), (a, b))

    return gemm(
        alpha=alpha,
        a=a,
        b=b,
        beta=beta,
        c=c,
        trans_a=trans_a,
        trans_b=trans_b,
        overwrite_c=overwrite_c,
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
