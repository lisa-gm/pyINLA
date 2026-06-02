# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Forked and modified from cupy.cublas.syrk: https://github.com/cupy/cupy/blob/3a2c950d64ee707096bc7ca1bf0b953a08206384/cupy/cublas.py#L930
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

def syherk(a, hw_target,c=None, alpha=1.0, beta=0.0, trans=0, lower=False, cu_chol=False):
    """Wrapper for the trsm function to call depending on wheter the solve happens on the host or the device
    
        For Compatibility this function accepts exactly the same parameters as what the scipy and cupy implementations accept
        plus the side parameter which can either be 0 or 1 for left or right hand side
    """
    
    if  hw_target == "host":
        return matmul_syherk_host(a, c, alpha, beta, trans, lower)
    elif hw_target == "accelerator":
        return matmul_syherk_accelerator(a, trans, c, alpha, beta, lower, cu_chol)
    else:
        ModuleNotFoundError("Unknown Module")

def matmul_syherk_host(a, c=None, alpha=1.0, beta=1.0, trans=0, lower=False,
                     overwrite_c=False, check_finite=True,):
    """Computes out = alpha * op(a) @ op(a)^T + beta * b

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'C'.
    """

    a1 = _asarray_validated(a, check_finite=check_finite)
    if c is None:
        c1 = None
    else:
        c1 = _asarray_validated(c, check_finite=check_finite)
    
    overwrite_c = overwrite_c or _datacopied(c1, c)

    x = _syherk(a1, c1, alpha, beta, trans, lower, overwrite_c)
    return x


# syherk without the input validation
def _syherk(a1, c1=None, alpha=1.0, beta=0.0, trans=0, lower=False,
                      overwrite_c=False):

    trans = {'N': 0, 'T': 1, 'C': 2}.get(trans, trans)

    if np.iscomplexobj(a1):
        syherk = get_blas_funcs(('herk'), (a1, a1))
    else:
        syherk = get_blas_funcs(('syrk'), (a1, a1))

    out = syherk(alpha, a1, beta, c1, trans, lower, overwrite_c)

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

def matmul_syherk_accelerator(a, trans='N', out=None, alpha=1.0, beta=0.0, lower=False, cu_chol=False):
    """Computes out := alpha*op1(a)*op2(a) + beta*out

    op1(a) = a if trans is 'N', op2(a) = a.T if transa is 'N'
    op1(a) = a.T if trans is 'T', op2(a) = a if transa is 'T'
    lower specifies  whether  the  upper  or  lower triangular
    part  of the  array  out  is to be  referenced
    """
    assert a.ndim == 2
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.ssyrk
    elif dtype == 'd':
        func = cublas.dsyrk
    elif dtype == 'F':
        try:
            func = cublas.cherk
        except(AttributeError):
            out = gemm(a, a, out, trans_b='C', alpha=alpha, beta=beta)
            return out
    elif dtype == 'D':
        try:
            func = cublas.zherk
        except(AttributeError):
            out = gemm(a, a, out, trans_b='C', alpha=alpha, beta=beta)
            return out
    else:
        raise TypeError('invalid dtype')
    
    # If this is run in combination with cholesky, it will be necessary to flip lower
    if cu_chol:
        lower = not lower

    trans = _trans_to_cublas_op(trans)
    if trans == cublas.CUBLAS_OP_N:
        n, k = a.shape
    else:
        k, n = a.shape
    if out is None:
        out = cp.zeros((n, n), dtype=dtype, order='F')
        beta = 0.0
    else:
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
            a = a.copy(order='C')
            trans = 1 - trans
            lda = a.shape[1]
        try:
            func(handle, 1 - uplo, trans, n, k,
                 alpha_ptr, a.data.ptr, lda,
                 beta_ptr, out.data.ptr, ldo)
        finally:
            cublas.setPointerMode(handle, orig_mode)

    else:
        if not a._f_contiguous:
            a = a.copy(order='F')
            lda = a.shape[0]
            trans = 1 - trans
        c = out
        if not out._f_contiguous:
            c = out.copy(order='F')
        try:
            func(handle, uplo, trans, n, k,
                 alpha_ptr, a.data.ptr, lda,
                 beta_ptr, out.data.ptr, ldo)
        finally:
            cublas.setPointerMode(handle, orig_mode)
        if not out._f_contiguous:
            out[...] = c
    return out