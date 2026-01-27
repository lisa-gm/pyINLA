import numpy as np
import scipy.sparse as sp

from .operations import Operation
from .matmul import dispatch_matmul

# At module level
_OPERATION_MAP = {
    Operation.MATMUL: dispatch_matmul,
}


def blas_dispatch(operation: Operation, left, right):
    # Type checking
    left_type = _get_matrix_type(left)
    right_type = _get_matrix_type(right)

    # Dispatch based on operation
    dispatch_func = _OPERATION_MAP[operation]
    return dispatch_func(left, right, left_type, right_type)


def _get_matrix_type(data):
    """Determine the type of matrix data"""
    if sp.issparse(data):
        return "sparse"
    if isinstance(data, np.ndarray):
        return "dense"
    raise TypeError(f"Unknown matrix type: {type(data)}")
