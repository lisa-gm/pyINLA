# src/dalia/backend/datastructures/matrix/core/utils.py
import numpy as np
import scipy.sparse as sp
from dalia import cupy_version, target_list


if cupy_version is not None:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp


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
    if cupy_version is not None:
        if isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        if cu_sp.issparse(data):
            return cp.asnumpy(data.toarray())
    return np.asarray(data)  # Works for arrays and views


def tohost(data):
    """Transfer data from accelerator to host"""
    return data.get()


def toaccelerator(data):
    """Transfer data from host to accelerator"""
    if sp.issparse(data):
        return cu_sp.csr_matrix(data)
    return cp.asarray(data)


def settarget (data, hw_target):
    """Set the hardware target for the data, transferring it if necessary"""
    if hw_target is not None and hw_target not in target_list:
        raise ValueError(f"Invalid hardware target type '{hw_target}'. Supported target types are {target_list}.")
    if hw_target is not None:
        if hw_target == 'accelerator' and 'cupy' not in str(type(data)):
            data = toaccelerator(data)
        elif hw_target == 'host' and 'cupy' in str(type(data)):
            data = tohost(data)
    else:
        # Infer hardware target from data type
        if 'cupy' in str(type(data)):
            hw_target = 'accelerator'
        else:
            hw_target = 'host'

    return data, hw_target