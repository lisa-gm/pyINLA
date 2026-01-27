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
