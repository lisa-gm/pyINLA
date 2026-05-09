# src/dalia/backend/datastructures/matrix/dispatch/dispatcher.py
import numpy as np
import scipy.sparse as sp

from dalia import cupy_version
# TODO: Change this to use flags instead of try
if cupy_version is not None:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp


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
    left_type, left_hw_target = _get_matrix_type(left)
    right_type, right_hw_target = _get_matrix_type(right)
    print(left_hw_target, right_hw_target)
    if left_hw_target != right_hw_target:
        # Handle hw_target mismatch
        # TODO: Make this work for different aproaches
        right = _hw_target_handler(right, right_hw_target, right_type)

    print(left_hw_target, right_hw_target)
    # Dispatch based on operation
    dispatch_func = _OPERATION_MAP[operation]
    print()
    return dispatch_func(left, right, left_type, right_type, left_hw_target)

def _hw_target_handler(data, hw_target, matrix_type):
    # Moves data to other hw_target
    if hw_target == "host":
        if matrix_type == "sparse":
            return cu_sp.csr_matrix(data)
        if matrix_type == "dense":
            return cp.asarray(data)
    if hw_target == "accelerator":
        if matrix_type == "sparse":
            return data.get()
        if matrix_type == "dense":
            return data.get()
    raise TypeError(f"Unknown hw_target type: {hw_target}")

def _get_matrix_type(data):
    """Determine the type of matrix data"""
    if sp.issparse(data):
        return "sparse", "host"
    if isinstance(data, np.ndarray):
        return "dense", "host"
    if cupy_version is not None:
         if cu_sp.issparse(data):
            return "sparse", "accelerator"
         if isinstance(data, cp.ndarray):
            return "dense", "accelerator"
    raise TypeError(f"Unknown matrix type: {type(data)}")

