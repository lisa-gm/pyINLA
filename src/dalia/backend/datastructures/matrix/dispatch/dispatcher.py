# src/dalia/backend/datastructures/matrix/dispatch/dispatcher.py
import numpy as np
import scipy.sparse as sp

from .add import dispatch_add
from .matmul import dispatch_matmul
from .operations import Operation
from .sub import dispatch_sub

# At module level
_OPERATION_MAP = {
    Operation.MATMUL: dispatch_matmul,
    Operation.ADD: dispatch_add,
    Operation.SUB: dispatch_sub,
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
