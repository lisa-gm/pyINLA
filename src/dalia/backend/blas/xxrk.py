"""


Credits:
- The host-side was forked and adapted from scipy.linal.solve_triangular: https://github.com/scipy/scipy/blob/v1.15.3/scipy/linalg/_basic.py#L411
- The Nvidia accelerator-side was forked and adapted from cupy.cublas.syrk: https://github.com/cupy/cupy/blob/3a2c950d64ee707096bc7ca1bf0b953a08206384/cupy/cublas.py#L930

"""

from typing import Literal

import numpy as np
from scipy.linalg._decomp import _asarray_validated
from scipy.linalg._misc import _datacopied
from scipy.linalg.blas import get_blas_funcs

from dalia.backend.config import cupy_version, nvmath_version
# from dalia.backend.datastructures import Matrix

from .gemm import matmul_gemm_accelerator

if cupy_version is not None:
    import cupy as cp
    from cupy import _core
    from cupy.cuda import device
    from cupy_backends.cuda.libs import cublas

if nvmath_version is not None:
    from nvmath.bindings import cublas as nvcublas


# def xxrk(
#     uplo: Literal["U", "u", "L", "l"],
#     trans_a: Literal["N", "n", "T", "t", "C", "c"],
#     alpha: float,
#     a: Matrix,
#     beta: float,
#     c: Matrix = None,
#     hw_target: Literal["default", "host", "accelerator"] = "default",
# ) -> Matrix | None:
#     """Wrapper for performing symmetric (syrk) and hermitian (herk) rank-k updates on
#     Matrix datastructures.

#     This routine performs one of the following symmetric (hermitian) rank k operations:
#         C = alpha * op(A) @ op(A)^T/H + beta * C
#     or
#         C = alpha * op(A)^T/H @ op(A) + beta * C

#     Parameters
#     ----------
#     uplo : {'U', 'u', 'L', 'l'}
#         Specifies whether the upper or lower triangular part of the result is
#         to be referenced. 'U' or 'u' for upper, 'L' or 'l' for lower.
#     trans_a : {'N', 'n', 'T', 't', 'C', 'c'}
#         Specifies the operation to be performed on matrix `a`. 'N' or 'n' for
#         no transpose, 'T' or 't' or 'C' or 'c' for transpose/conjugate transpose.
#     alpha : float
#         Scalar to be multiplied with matrix `a`.
#     a : Matrix
#         Matrix to be rank-updated.
#     beta : float
#         Scalar to be multiplied with matrix `c`.
#     c : Matrix, optional
#         Output matrix that will be added to the result. If None, a new matrix will
#         be created, otherwise the matrix c will be overwritten with the result.
#     hw_target : {'default', 'host', 'accelerator'}, default='default'
#         Hardware target, either "host" or "accelerator" depending on the current
#         location of matrix `a`. If set to "default", the function will automatically
#         determine the hardware target based on the location of matrix `a`.

#     Returns
#     -------
#     Matrix or None
#         Resulting matrix after the rank-k update if `c` is not provided. If `c`
#         is provided, it will be overwritten in-place with the result, and the
#         routine will return None.
#     """

#     if hw_target == "host":
#         return _xxrk_host(a, c, alpha, beta, trans_a, lower, overwrite_c)
#     elif hw_target == "accelerator":
#         return _xxrk_accelerator(a, c, alpha, beta, trans_a, lower, overwrite_c)
#     else:
#         ModuleNotFoundError("Unknown Module")


def xxrk(a, hw_target,c=None, alpha=1.0, beta=0.0, trans_a=0, lower=0, overwrite_c=0):
    """Wrapper for the SYRK and HERK function to call depending on wheter the operation happens on the host or the device

        Computes out = alpha * op(a) @ op(a)^T + beta * b

        op(a) = a if trans is 'N', op(a) = a.T if trans is 'T',
        op(a) = a.T.conj() if trans is 'C'.

        Args:
            a:              Matrix to be rank-updated
            hw_target:      Hardware target, either "host" or "accelerator" depending on the current location of a
            c:              Matrix that will be added to the result
            alpha:          Scalar to be multiplied with a
            beta:           Scalar to be multiplied with c
            trans_a:          {'N','T','C'} or {'0','1','2'} respectively determines op(a)
            lower:          Bool determining wheter the upper or lower result should be referenced
            overwrite_c:    Bool determining wheter the result should overwrite Matrix c

        Returns:
            out:            Resulting Matrix

        Raises:
            ModuelNotFoundError:    If the hw_target is not in {"host","accelerator"}
            TypeError:              If the Matrix has an invalid dtype or another parameter cannot be recognized
    """

    if  hw_target == "host":
        return _xxrk_host(a, c, alpha, beta, trans_a, lower, overwrite_c)
    elif hw_target == "accelerator":
        return _xxrk_accelerator(a, c, alpha, beta, trans_a, lower, overwrite_c)
    else:
        ModuleNotFoundError("Unknown Module")


# Host-side Kernels
def _xxrk_host(
    a,
    c=None,
    alpha=1.0,
    beta=1.0,
    trans=0,
    lower=False,
    overwrite_c=False,
    check_finite=True,
):
    """Computes SYRK and HERK on the host

    additional Argument check_finite that checks if a is finite
    """

    a1 = _asarray_validated(a, check_finite=check_finite)
    if c is None:
        c1 = None
    else:
        c1 = _asarray_validated(c, check_finite=check_finite)

    overwrite_c = overwrite_c or _datacopied(c1, c)

    x = _xxrk(a1, c1, alpha, beta, trans, lower, overwrite_c)
    return x


def _xxrk(a1, c1=None, alpha=1.0, beta=0.0, trans=0, lower=False, overwrite_c=False):
    """xxrk without the input validation"""

    trans = {"N": 0, "T": 1, "C": 2}.get(trans, trans)

    if np.iscomplexobj(a1):
        xxrk = get_blas_funcs(("herk"), (a1, a1))
    else:
        xxrk = get_blas_funcs(("syrk"), (a1, a1))

    out = xxrk(alpha, a1, beta, c1, trans, lower, overwrite_c)

    return out


# Nvidia Accelerator-side Kernels
def _trans_to_cublas_op(trans):
    """Util functions for cuda xxrk"""
    if trans == "N" or trans == cublas.CUBLAS_OP_N:
        trans = cublas.CUBLAS_OP_N
    elif trans == "T" or trans == cublas.CUBLAS_OP_T:
        trans = cublas.CUBLAS_OP_T
    elif trans == "C" or trans == cublas.CUBLAS_OP_C:
        trans = cublas.CUBLAS_OP_C
    else:
        raise TypeError("invalid trans (actual: {})".format(trans))
    return trans


def _decide_ld_and_trans(a, trans):
    ld = None
    if trans in (cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T):
        if a._f_contiguous:
            ld = a.shape[0]
        elif a._c_contiguous:
            ld = a.shape[1]
            trans = 1 - trans
    return ld, trans


def _get_scalar_ptr(a, dtype):
    if isinstance(a, cp.ndarray):
        if a.dtype != dtype:
            a = cp.array(a, dtype=dtype)
        a_ptr = a.data.ptr
    else:
        if not (isinstance(a, np.ndarray) and a.dtype == dtype):
            a = np.array(a, dtype=dtype)
        a_ptr = a.ctypes.data
    return a, a_ptr


def _xxrk_accelerator(
    a, c=None, alpha=1.0, beta=0.0, trans="N", lower=False, overwrite_c=0
):
    """Computes SYRK and HERK on a cuda accelerator

    if nvmath is not installed HERK will call GEMM instead
    """
    assert a.ndim == 2
    dtype = a.dtype.char
    if dtype == "f":
        func = cublas.ssyrk
    elif dtype == "d":
        func = cublas.dsyrk
    elif dtype == "F":
        try:
            func = cublas.cherk
        except AttributeError:
            if nvmath_version is not None:
                func = nvcublas.cherk
            else:
                matmul_gemm_accelerator(a, a, out, trans_b="C", alpha=alpha, beta=beta)
    elif dtype == "D":
        try:
            func = cublas.zherk
        except AttributeError:
            if nvmath_version is not None:
                func = nvcublas.zherk
            else:
                matmul_gemm_accelerator(a, a, out, trans_b="C", alpha=alpha, beta=beta)
    else:
        raise TypeError("invalid dtype")

    trans = _trans_to_cublas_op(trans)
    if trans == cublas.CUBLAS_OP_N:
        n, k = a.shape
    else:
        k, n = a.shape
    out = None
    if c is None:
        out = cp.zeros((n, n), dtype=dtype, order="F")
        beta = 0.0
    else:
        if overwrite_c:
            out = c
        else:
            out = c.copy(order="F")
        assert out.ndim == 2
        assert out.shape == (n, n)
        assert out.dtype == dtype

    if lower:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
    beta, beta_ptr = _get_scalar_ptr(beta, a.dtype)
    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    if isinstance(alpha, cp.ndarray) or isinstance(beta, cp.ndarray):
        if not isinstance(alpha, cp.ndarray):
            alpha = cp.array(alpha)
            alpha_ptr = alpha.data.ptr
        if not isinstance(beta, cp.ndarray):
            beta = cp.array(beta)
            beta_ptr = beta.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    lda, trans = _decide_ld_and_trans(a, trans)
    ldo, _ = _decide_ld_and_trans(out, trans)

    if out._c_contiguous:
        if not a._c_contiguous:
            a = a.copy(order="C")
            trans = 1 - trans
            lda = a.shape[1]
        try:
            func(
                handle,
                1 - uplo,
                trans,
                n,
                k,
                alpha_ptr,
                a.data.ptr,
                lda,
                beta_ptr,
                out.data.ptr,
                ldo,
            )
        finally:
            cublas.setPointerMode(handle, orig_mode)

    else:
        if not a._f_contiguous:
            a = a.copy(order="F")
            lda = a.shape[0]
            trans = 1 - trans
        c = out
        if not out._f_contiguous:
            c = out.copy(order="F")
        try:
            func(
                handle,
                uplo,
                trans,
                n,
                k,
                alpha_ptr,
                a.data.ptr,
                lda,
                beta_ptr,
                out.data.ptr,
                ldo,
            )
        finally:
            cublas.setPointerMode(handle, orig_mode)
        if not out._f_contiguous:
            out[...] = c
    return out
