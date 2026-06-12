# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Forked and modified from cupy.cublas.gemm: https://github.com/cupy/cupy/blob/3a2c950d64ee707096bc7ca1bf0b953a08206384/cupy/cublas.py#L689
# and scipy.linal.solve_triangular: https://github.com/scipy/scipy/blob/v1.15.3/scipy/linalg/_basic.py#L411

import numpy as np

from scipy.linalg.blas import get_blas_funcs
from scipy.linalg._misc import _datacopied
from scipy.linalg._decomp import _asarray_validated


from dalia.backend.config import cupy_version

if cupy_version is not None:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas
    from cupy import _core
    from cupy.cuda import device


def gemm (a, b, hw_target, c=None, alpha=1.0, beta=0.0, trans_a ='N', trans_b ='N', overwrite_c = 0):
    """Wrapper to call GEMM for host or device
    
    Computes out = alpha * op(a) @ op(b) + beta * c

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'C'.
    op(b) = b if transb is 'N', op(b) = b.T if transb is 'T',
    op(b) = b.T.conj() if transb is 'C'.

    Args:
            a:              Left Matrix
            b:              Right Matrix
            hw_target:      Hardware target, either "host" or "accelerator" depending on the current location of a
            c:              Matrix that will be added to the result
            alpha:          Scalar to be multiplied with a
            beta:           Scalar to be multiplied with c
            trans_a:        {'N','T','C'} or {'0','1','2'} respectively determines op(a)
            trans_b:        {'N','T','C'} or {'0','1','2'} respectively determines op(b)
            overwrite_c:    Bool determining wheter the result should overwrite Matrix c
        
        Returns:
            out:            Resulting Matrix

        Raises:
            ModuelNotFoundError:    If the hw_target is not in {"host","accelerator"}
            TypeError:              If the Matrix has an invalid dtype or another parameter cannot be recognized
    """
    

    if hw_target == "host":
        return matmul_gemm_host(a, b, c, alpha, beta, trans_a, trans_b, overwrite_c)
    elif hw_target == "accelerator":
        return matmul_gemm_accelerator(a, b, c, alpha, beta, trans_a, trans_b, overwrite_c)
    else:
        ModuleNotFoundError("Unknown Module")


def matmul_gemm_host(a, b, c=None, alpha=1.0, beta=0.0, trans_a=0, trans_b=0, overwrite_c=0, check_finite=False):
    """Computes GEMM on the host

    additional Argument check_finite that checks if a and b are finite
    """

    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    if c is None:
        c1 = None
    else:
        c1 = _asarray_validated(c, check_finite=check_finite)

    transa = True
    transb = True
    if trans_a == 'N':
        transa = False
    if trans_b == 'N':
        transb = False

    if not transa and not transb:
        if a1.shape[1] != b1.shape[0]:
            raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible (1,0)')
        
    elif transa and not transb:
        if a1.shape[0] != b1.shape[0]:
            raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible (0,0)')
        
    elif not transa and transb:
        if a1.shape[1] != b1.shape[1]:
            raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible (1,1)')
        
    else:
        if a1.shape[0] != b1.shape[1]:
            raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible (0,1)')
    
    if beta != 0 and c1 is None:
        raise ValueError('expected C matrix')

    # accommodate empty arrays
    if b1.size == 0:
        dt_nonempty = matmul_gemm_host(
            np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)
        ).dtype
        return np.empty_like(b1, dtype=dt_nonempty)
    
    if c1 is not None:
        overwrite_c = overwrite_c or _datacopied(c1, c)
    
    x = _matmul_gemm(a1, b1, alpha, beta, c1, trans_a, trans_b, overwrite_c)
    return x


# gemm without the input validation
def _matmul_gemm(a1, b1, alpha=1.0, beta=0.0, c1=None, trans_a=0, trans_b=0, overwrite_c=0):

    trans_a = {'N': 0, 'T': 1, 'C': 2}.get(trans_a, trans_a)
    trans_b = {'N': 0, 'T': 1, 'C': 2}.get(trans_b, trans_b)
    gemm, = get_blas_funcs(('gemm',), (a1, b1))

    if beta == 0:
        out = gemm(alpha, a1, b1, beta=beta, trans_a=trans_a, trans_b=trans_b, overwrite_c=overwrite_c)
    else:
        out = gemm(alpha, a1, b1, beta, c1, trans_a, trans_b, overwrite_c)
    

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

# TODO: warnings for copies, maybe...
def matmul_gemm_accelerator(a, b, c=None, alpha=1.0, beta=0.0, transa=0, transb=0, overwrite_c=0):
    """Computes GEMM on a cuda accelerator"""
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sgemm
    elif dtype == 'd':
        func = cublas.dgemm
    elif dtype == 'F':
        func = cublas.cgemm
    elif dtype == 'D':
        func = cublas.zgemm
    else:
        raise TypeError('invalid dtype')
    

    transa = _trans_to_cublas_op(transa)
    transb = _trans_to_cublas_op(transb)
    if transa == cublas.CUBLAS_OP_N:
        m, k = a.shape
    else:
        k, m = a.shape
    if transb == cublas.CUBLAS_OP_N:
        n = b.shape[1]
        assert b.shape[0] == k
    else:
        n = b.shape[0]
        assert b.shape[1] == k

    out = None
    if c is None:
        out = cp.empty((m, n), dtype=dtype, order='F')
        beta = 0.0
    else:
        if overwrite_c:
            out = c
        else:
            out = c.copy(order='F')
        assert out.ndim == 2
        assert out.shape == (m, n)
        assert out.dtype == dtype

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

    lda, transa = _decide_ld_and_trans(a, transa)
    ldb, transb = _decide_ld_and_trans(b, transb)
    if not (lda is None or ldb is None):
        if out._f_contiguous:
            try:
                func(handle, transa, transb, m, n, k, alpha_ptr,
                     a.data.ptr, lda, b.data.ptr, ldb, beta_ptr, out.data.ptr,
                     m)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out
        elif out._c_contiguous:
            # Computes out.T = alpha * b.T @ a.T + beta * out.T
            try:
                func(handle, 1 - transb, 1 - transa, n, m, k, alpha_ptr,
                     b.data.ptr, ldb, a.data.ptr, lda, beta_ptr, out.data.ptr,
                     n)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out

    a, lda = _change_order_if_necessary(a, lda)
    b, ldb = _change_order_if_necessary(b, ldb)
    c = out
    if not out._f_contiguous:
        c = out.copy(order='F')
    try:
        func(handle, transa, transb, m, n, k, alpha_ptr, a.data.ptr, lda,
             b.data.ptr, ldb, beta_ptr, c.data.ptr, m)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if not out._f_contiguous:
        _core.elementwise_copy(c, out)
    return out