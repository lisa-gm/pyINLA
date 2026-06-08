# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Forked and modified from cupy.cublas.gemm: https://github.com/cupy/cupy/blob/3a2c950d64ee707096bc7ca1bf0b953a08206384/cupy/cublas.py#L689
# and scipy.linal.solve_triangular: https://github.com/scipy/scipy/blob/v1.15.3/scipy/linalg/_basic.py#L411

import numpy as np

from scipy.linalg.blas import get_blas_funcs
from scipy.linalg._misc import _datacopied
from scipy.linalg._decomp import _asarray_validated


from dalia.backend.config import cupy_version, nvmath_version
from .gemm import matmul_gemm_accelerator

# TODO: Change this to use flags instead of try
if cupy_version is not None:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas
    from cupy import _core
    from cupy.cuda import device

if nvmath_version is not None:
    from nvmath.bindings import cublas as nvcublas


def trmm (a, b, hw_target, alpha=1.0, side=0, lower=0, trans_a ='N', diag=0, overwrite_b=0):
    """Wrapper to call GeMM for host or device"""
    

    if hw_target == "host":
        return matmul_trmm_host(a, b, alpha, trans_a, side, lower, diag, overwrite_b)
    elif hw_target == "accelerator":
        return matmul_trmm_accelerator(a, b, alpha, trans_a, side, lower, diag, overwrite_b)
    else:
        ModuleNotFoundError("Unknown Module")


def matmul_trmm_host(a, b, alpha=1.0, trans_a=0, side=0, lower=0, diag=0, overwrite_b=0, check_finite=False):
    """Computes out = alpha * op(a) @ b or alpha * b @ op(a) where a is a triangluar matrix

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T' or 'C',
    op(a) = a.T.conj() if transa is 'C'.

    side determines if a is on the left or on the right
    lower determines if a is an upper or lower triangular matrix
    diag determines if a is unit triangluar
    overwrite_b determines if out will overwrite b
    """

    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)


    if a1.shape[0] != a1.shape[1]:
        raise ValueError(f'a {a1.shape} is not a square matrix')
    if not side:
        if a1.shape[0] != b1.shape[0]:
            raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible')
    else:
        if a1.shape[0] != b1.shape[1]:
            raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible')

    # accommodate empty arrays
    if b1.size == 0:
        dt_nonempty = matmul_trmm_host(
            np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)
        ).dtype
        return np.empty_like(b1, dtype=dt_nonempty)
    
    x = _matmul_trmm(a1, b1, alpha, side, lower, trans_a, diag, overwrite_b)
    return x


# trmm without the input validation
def _matmul_trmm(a1, b1, alpha=1.0, side=0, lower=0, trans_a=0, diag=0, overwrite_b=0):

    trans_a = {'N': 0, 'T': 1, 'C': 2}.get(trans_a, trans_a)
    trmm, = get_blas_funcs(('trmm',), (a1, b1))

    
    out = trmm(alpha, a1, b1, side, lower, trans_a, diag, overwrite_b)
    

    return out


# Util functions for cupy gemm
def _trans_to_cublas_op(trans):
    if trans == 'N' or trans == cublas.CUBLAS_OP_N:
        trans = cublas.CUBLAS_OP_N
    elif trans == 'T' or trans == cublas.CUBLAS_OP_T:
        trans = cublas.CUBLAS_OP_T
    elif trans == 'C' or trans == cublas.CUBLAS_OP_C:
        trans = cublas.CUBLAS_OP_C
    else:
        raise TypeError('invalid trans (actual: {})'.format(trans))
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


def _change_order_if_necessary(a, lda):
    if lda is None:
        lda = a.shape[0]
        if not a._f_contiguous:
            a = a.copy(order='F')
    return a, lda

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
# Util functions for cupy gemm end


def matmul_trmm_accelerator(a, b, alpha=1.0, transa=0, side=0, lower=0, diag=0, overwrite_b=0):
    """Computes out = alpha * op(a) @ b or alpha * b @ op(a) where a is a triangular matrix

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T' or 'C',
    op(a) = a.T.conj() if transa is 'C'.

    side determines if a is on the left or on the right
    lower determines if a is an upper or lower triangular matrix
    diag determines if a is unit triangluar
    overwrite_b determines if out will overwrite b
    """
    if nvmath_version is not None:
        matmul_gemm_accelerator(a, b, alpha=alpha, transa=transa, transb="N")

    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype
    dtype = a.dtype.char
    if dtype == 'f':
        func = nvcublas.strmm
    elif dtype == 'd':
        func = nvcublas.dtrmm
    elif dtype == 'F':
        func = nvcublas.ctrmm
    elif dtype == 'D':
        func = nvcublas.ztrmm
    else:
        raise TypeError('invalid dtype')

    transa = _trans_to_cublas_op(transa)
    assert a.shape[0] == a.shape[1]
    lda = a.shape[0]
    m, n = b.shape
    ldb = m
    out = None
    if overwrite_b:
        out = b
        assert out.ndim == 2
        assert out.shape == (m, n)
        assert out.dtype == dtype
    else:
        out = cp.zeros((m, n), dtype=dtype, order='F')
    if a._c_contiguous:
        a = a.copy(order='F')
    if b._c_contiguous:
        b = b.copy(order='F')
    if lower:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if side:
        side = cublas.CUBLAS_SIDE_RIGHT
        assert lda == n
    else:
        side = cublas.CUBLAS_SIDE_LEFT
        assert lda == m

    if diag:
        diag = cublas.CUBLAS_DIAG_UNIT
    else:
        diag = cublas.CUBLAS_DIAG_NON_UNIT

    alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    if isinstance(alpha, cp.ndarray):
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    c = out
    if not out._f_contiguous:
        c = out.copy(order='F')
    try:
        func(handle, side, uplo, transa, diag, m, n, alpha_ptr, a.data.ptr, lda,
             b.data.ptr, ldb, c.data.ptr, m)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if not out._f_contiguous:
        _core.elementwise_copy(c, out)
    return out