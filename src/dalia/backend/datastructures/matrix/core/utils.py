# src/dalia/backend/datastructures/matrix/core/utils.py
import numpy as np
import scipy.sparse as sp


def wrap_result(data):
    """Wrap the result data in the appropriate Matrix subclass"""
    # pylint: disable=import-outside-toplevel
    from .dense import DenseMatrix
    from .sparse import SparseMatrix

    if isinstance(data, np.ndarray):
        return DenseMatrix(data)
    if sp.issparse(data):
        return SparseMatrix(data)
    raise TypeError(f"Unknown matrix type: {type(data)}")


def toarray(data):
    """Convert data to a dense numpy array"""
    if sp.issparse(data):
        return data.toarray()
    return np.asarray(data)  # Works for arrays and views
