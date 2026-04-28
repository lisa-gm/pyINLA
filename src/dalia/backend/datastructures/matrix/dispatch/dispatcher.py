# src/dalia/backend/datastructures/matrix/dispatch/dispatcher.py
import numpy as np
import scipy.sparse as sp
# TODO: Change this to use flags instead of try
try:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp
except ImportError:
    pass

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
    left_type, left_device = _get_matrix_type(left)
    right_type, right_device = _get_matrix_type(right)

    if left_device != right_device:
        # Handle device mismatch
        # TODO: Make this work for different aproaches
        right = _device_handler(right, right_device, right_type)

    # Dispatch based on operation
    dispatch_func = _OPERATION_MAP[operation]
    return dispatch_func(left, right, left_type, right_type)

def _device_handler(data, device, type):
    # Moves data to other device
    if device == "cpu":
        if type == "sparse":
            return cu_sp.csr_matrix(data)
        if type == "dense":
            return cp.asarray(data)
    if device == "gpu":
        if type == "sparse":
            return data.get()
        if type == "dense":
            return data.get()
    raise TypeError(f"Unknown device type: {device}")

def _get_matrix_type(data):
    """Determine the type of matrix data"""
    if sp.issparse(data):
        return "sparse", "cpu"
    if isinstance(data, np.ndarray):
        return "dense", "cpu"
    if cu_sp.issparse(data):
        return "sparse", "gpu"
    if isinstance(data, cp.ndarray):
        return "dense", "gpu"
    raise TypeError(f"Unknown matrix type: {type(data)}")

