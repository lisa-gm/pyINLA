# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Forked and modified from cupyx.linalg.solve_triangular: https://github.com/cupy/cupy/blob/3a2c950d64ee707096bc7ca1bf0b953a08206384/cupyx/scipy/linalg/_solve_triangular.py#L12
# and scipy.linal.solve_triangular: https://github.com/scipy/scipy/blob/v1.15.3/scipy/linalg/_basic.py#L411

import numpy as np
import math as math

from scipy.linalg.blas import get_blas_funcs
from scipy.linalg._misc import _datacopied
from scipy.linalg._decomp import _asarray_validated

from dalia.backend.config import cupy_version, target_list

if cupy_version is not None:
    import cupy as cp
    from cupy.cuda import cublas
    from cupy.cuda import device
    from cupy.linalg import _util


def trsm(a, b, hw_target, trans=0, lower = False, unit_diagonal=False,
                            overwrite_b=True, check_finite=False, side=0):
    """Wrapper for the trsm function to call depending on wheter the solve happens on the host or the device
    
        For Compatibility this function accepts exactly the same parameters as what the scipy and cupy implementations accept
        plus the side parameter which can either be 0 or 1 for left or right hand side
    """
    if hw_target == "host":
        return solve_triangular_host(a, b, trans, lower, unit_diagonal, overwrite_b, check_finite, side)
    elif hw_target == "accelerator":
        return solve_triangular_accelerator(a, b, trans, lower, unit_diagonal, overwrite_b, check_finite, side)
    else:
        raise ValueError(f"Invalid hardware target type '{hw_target}'. Supported target types are {target_list}.")
    


def solve_triangular_accelerator(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=False, side=0):
    """Solve the equation a x = b for x, assuming a is a triangular matrix.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(M,)`` or
            ``(M, N)``.
        lower (bool): Use only data contained in the lower triangle of ``a``.
            Default is to use upper triangle.
        trans (0, 1, 2, 'N', 'T' or 'C'): Type of system to solve:

            - *'0'* or *'N'* -- :math:`a x  = b`
            - *'1'* or *'T'* -- :math:`a^T x = b`
            - *'2'* or *'C'* -- :math:`a^H x = b`

        unit_diagonal (bool): If ``True``, diagonal elements of ``a`` are
            assumed to be 1 and will not be referenced.
        overwrite_b (bool): Allow overwriting data in b (may enhance
            performance)
        check_finite (bool): Whether to check that the input matrices contain
            only finite numbers. Disabling may give a performance gain, but may
            result in problems (crashes, non-termination) if the inputs do
            contain infinities or NaNs.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M,)`` or ``(M, N)``.

    .. seealso:: :func:`scipy.linalg.solve_triangular`
    """

    _util._assert_cupy_array(a, b)

    if a.ndim == 2:
        if a.shape[0] != a.shape[1]:
            raise ValueError('expected square matrix')
        if len(a) != len(b):
            raise ValueError('incompatible dimensions')
        batch_count = 0
    elif a.ndim > 2:
        if a.shape[-1] != a.shape[-2]:
            raise ValueError('expected a batch of square matrices')
        if a.shape[:-2] != b.shape[:a.ndim - 2]:
            raise ValueError('incompatible batch count')
        if b.ndim < a.ndim - 1 or a.shape[-2] != b.shape[a.ndim - 2]:
            raise ValueError('incompatible dimensions')
        batch_count = math.prod(a.shape[:-2])
    else:
        raise ValueError(
            'expected one square matrix or a batch of square matrices')

    # Cast to float32 or float64
    if a.dtype.char in 'fd':
        dtype = a.dtype
    else:
        dtype = np.promote_types(a.dtype.char, 'f')

    if check_finite:
        if a.dtype.kind == 'f' and not cp.isfinite(a).all():
            raise ValueError(
                'array must not contain infs or NaNs')
        if b.dtype.kind == 'f' and not cp.isfinite(b).all():
            raise ValueError(
                'array must not contain infs or NaNs')

    if batch_count:
        m, n = b.shape[-2:] if b.ndim == a.ndim else (b.shape[-1], 1)

        a_new_shape = (batch_count, m, m)
        b_shape = b.shape
        b_data_ptr = b.data.ptr
        # trsm receives Fortran array, but we want zero copy
        if trans == 'N' or trans == cublas.CUBLAS_OP_N:
            # normal Fortran upper == transpose C lower
            trans = cublas.CUBLAS_OP_T
            lower = not lower
            a = cp.ascontiguousarray(a.reshape(*a_new_shape), dtype=dtype)
        elif trans == 'T' or trans == cublas.CUBLAS_OP_T:
            # transpose Fortran upper == normal C lower
            trans = cublas.CUBLAS_OP_N
            lower = not lower
            a = cp.ascontiguousarray(a.reshape(*a_new_shape), dtype=dtype)
        elif trans == 'C' or trans == cublas.CUBLAS_OP_C:
            if dtype == 'f' or dtype == 'd':
                # real numbers
                # Hermitian Fortran upper == transpose Fortran upper
                #                         == normal C lower
                trans = cublas.CUBLAS_OP_N
                lower = not lower
                a = cp.ascontiguousarray(a.reshape(*a_new_shape),
                                           dtype=dtype)
            else:
                # complex numbers
                trans = cublas.CUBLAS_OP_C
                a = cp.ascontiguousarray(
                    a.reshape(*a_new_shape).transpose(0, 2, 1), dtype=dtype)
        else:  # know nothing about `trans`, just convert C to Fortran
            a = cp.ascontiguousarray(
                a.reshape(*a_new_shape).transpose(0, 2, 1), dtype=dtype)
        b = cp.ascontiguousarray(
            b.reshape(batch_count, m, n).transpose(0, 2, 1), dtype=dtype)
        if b.data.ptr == b_data_ptr and not overwrite_b:
            b = b.copy()

        start = a.data.ptr
        step = m * m * a.itemsize
        stop = start + step * batch_count
        a_array = cp.arange(start, stop, step, dtype=cp.uintp)

        start = b.data.ptr
        step = m * n * b.itemsize
        stop = start + step * batch_count
        b_array = cp.arange(start, stop, step, dtype=cp.uintp)
    else:
        a = cp.array(a, dtype=dtype, order='F', copy=None)
        b = cp.array(b, dtype=dtype, order='F',
                       copy=(None if overwrite_b else True))

        m, n = (b.size, 1) if b.ndim == 1 else b.shape

        if trans == 'N':
            trans = cublas.CUBLAS_OP_N
        elif trans == 'T':
            trans = cublas.CUBLAS_OP_T
        elif trans == 'C':
            trans = cublas.CUBLAS_OP_C

    cublas_handle = device.get_cublas_handle()
    one = np.array(1, dtype=dtype)

    if lower:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if unit_diagonal:
        diag = cublas.CUBLAS_DIAG_UNIT
    else:
        diag = cublas.CUBLAS_DIAG_NON_UNIT

    if side:
        side = cublas.CUBLAS_SIDE_RIGHT
    else:
        side = cublas.CUBLAS_SIDE_LEFT

    if batch_count:
        if dtype == 'f':
            trsm = cublas.strsmBatched
        elif dtype == 'd':
            trsm = cublas.dtrsmBatched
        elif dtype == 'F':
            trsm = cublas.ctrsmBatched
        else:  # dtype == 'D'
            trsm = cublas.ztrsmBatched
        trsm(
            cublas_handle, side, uplo,
            trans, diag,
            m, n, one.ctypes.data, a_array.data.ptr, m,
            b_array.data.ptr, m, batch_count)
        return b.transpose(0, 2, 1).reshape(b_shape)
    else:
        if dtype == 'f':
            trsm = cublas.strsm
        elif dtype == 'd':
            trsm = cublas.dtrsm
        elif dtype == 'F':
            trsm = cublas.ctrsm
        else:  # dtype == 'D'
            trsm = cublas.ztrsm
        trsm(
            cublas_handle, side, uplo,
            trans, diag,
            m, n, one.ctypes.data, a.data.ptr, m, b.data.ptr, m)
        return b
    

def solve_triangular_host(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=True, side=0):
    """
    Solve the equation ``a x = b`` for `x`, assuming a is a triangular matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        Right-hand side matrix in ``a x = b``
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.
    trans : {0, 1, 2, 'N', 'T', 'C'}, optional
        Type of system to solve:

        ========  =========
        trans     system
        ========  =========
        0 or 'N'  a x  = b
        1 or 'T'  a^T x = b
        2 or 'C'  a^H x = b
        ========  =========
    unit_diagonal : bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and
        will not be referenced.
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system ``a x = b``.  Shape of return matches `b`.

    Raises
    ------
    LinAlgError
        If `a` is singular

    Notes
    -----
    .. versionadded:: 0.9.0

    Examples
    --------
    Solve the lower triangular system a x = b, where::

             [3  0  0  0]       [4]
        a =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]

    >>> import numpy as np
    >>> from scipy.linalg import solve_triangular
    >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    >>> b = np.array([4, 2, 4, 2])
    >>> x = solve_triangular(a, b, lower=True)
    >>> x
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
    >>> a.dot(x)  # Check the result
    array([ 4.,  2.,  4.,  2.])

    """

    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)

    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    if a1.shape[0] != b1.shape[0]:
        raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible')

    # accommodate empty arrays
    if b1.size == 0:
        dt_nonempty = solve_triangular_host(
            np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)
        ).dtype
        return np.empty_like(b1, dtype=dt_nonempty)

    overwrite_b = overwrite_b or _datacopied(b1, b)

    x = _solve_triangular(a1, b1, trans, lower, unit_diagonal, overwrite_b, side)
    return x


# solve_triangular without the input validation
def _solve_triangular(a1, b1, trans=0, lower=False, unit_diagonal=False,
                      overwrite_b=False, side=0):

    trans = {'N': 0, 'T': 1, 'C': 2}.get(trans, trans)
    trsm, = get_blas_funcs(('trsm',), (a1, b1))

    alpha = 1.0

    if a1.flags.f_contiguous or trans == 2:
        x = trsm(alpha, a1, b1, overwrite_b=overwrite_b, lower=lower,
                        trans_a=trans, diag=unit_diagonal, side=side)
    else:
        # transposed system is solved since trsm expects Fortran ordering
        x = trsm(alpha, a1.T, b1, overwrite_b=overwrite_b, lower=not lower,
                        trans_a=not trans, diag=unit_diagonal, side=side)

    return x