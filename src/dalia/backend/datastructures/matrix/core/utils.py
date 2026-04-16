# src/dalia/backend/datastructures/matrix/core/utils.py
import numpy as np
import scipy.sparse as sp
# TODO: Change this to use flags instead of try
try:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp
except ImportError:
    pass

def wrap_result(data):
    """Wrap the result data in the appropriate Matrix subclass"""
    # pylint: disable=import-outside-toplevel
    from .dense import DenseMatrix
    from .sparse import SparseMatrix

    if isinstance(data, np.ndarray):
        return DenseMatrix(data)
    if sp.issparse(data):
        return SparseMatrix(data)
    if isinstance(data, cp.ndarray):
        return DenseMatrix(data)
    if cu_sp.issparse(data):
        return SparseMatrix(data)
    raise TypeError(f"Unknown matrix type: {type(data)}")


def toarray(data):
    """Convert data to a dense numpy array"""
    if sp.issparse(data):
        return data.toarray()
    if isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    if cu_sp.issparse(data):
        return cp.asnumpy(data.toarray())
    return np.asarray(data)  # Works for arrays and views

def tocpu(data):
    """Convert GPU data to a CPU array"""
    return data.get()

def togpu(data):
    """Convert CPU data to a GPU array"""
    if sp.issparse(data):
        if data.dtype.char not in '?fdFD': 
            # cupy sparse only supports bool, float32, float64, complex64, complex128
            # convert to float64 by default if unsupported dtype
            # Might act weird for non-numeric types
            return cu_sp.csr_matrix(data, dtype=cp.float64)
        return cu_sp.csr_matrix(data)
    return cp.asarray(data)
