# src/dalia/backend/datastructures/matrix/dispatch/dispatcher.py
from enum import Enum
from typing import Union

import numpy as np
import scipy.sparse as sp

from dalia.backend.config import (
    cupy_version,
    gputil_version,
    memory_regime,
    memory_threshold,
    regime_list,
)
from dalia.backend.datastructures.matrix.operators.add import dispatch_add
from dalia.backend.datastructures.matrix.operators.matmul import dispatch_matmul
from dalia.backend.datastructures.matrix.operators.mul import dispatch_mul
from dalia.backend.datastructures.matrix.operators.sub import dispatch_sub

if cupy_version is not None:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp

if gputil_version is not None:
    import GPUtil


class Operation(Enum):
    MUL = "mul"
    MATMUL = "matmul"
    ADD = "add"
    SUB = "sub"


_OPERATION_MAP = {
    Operation.MUL: dispatch_mul,
    Operation.MATMUL: dispatch_matmul,
    Operation.ADD: dispatch_add,
    Operation.SUB: dispatch_sub,
}


def dispatch(
    left_operand: Union[np.ndarray, sp.spmatrix],
    right_operand: Union[np.ndarray, sp.spmatrix],
    operation: Operation,
) -> Union[np.ndarray, sp.spmatrix]:
    """Dispatch matrix operations to suited backends based on the types
    of the `left_operand` and `right_operand` and their hardware targets.

    Parameters
    ----------
    left_operand : Union[np.ndarray, sp.spmatrix]
        The left operand for the matrix operation.
    right_operand : Union[np.ndarray, sp.spmatrix]
        The right operand for the matrix operation.
    operation : Operation
        The matrix operation to be performed (MUL, MATMUL, ADD, SUB).

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        The result of the matrix operation, dispatched to the appropriate backend.

    Raises
    ------
    TypeError
        If the operation cannot be performed on the given types of operands.
    ValueError
        If the memory regime is invalid or if there is an out-of-memory error in manual mode
    """
    # Type checking
    left_type, left_hw_target = _get_dispatch_metadata(data=left_operand)
    right_type, right_hw_target = _get_dispatch_metadata(data=right_operand)

    if left_hw_target != right_hw_target:
        # Handle hw_target mismatch
        # TODO: Make this work for different aproaches
        target = None
        if memory_regime == "auto":
            target = _target_decider(
                left_operand,
                right_operand,
                left_hw_target,
                right_hw_target,
                left_type,
                right_type,
            )

            if left_hw_target != target:
                left_operand = _hw_target_handler(left_operand, target, left_type)
                left_hw_target = target
            if right_hw_target != target:
                right_operand = _hw_target_handler(right_operand, target, right_type)
                right_hw_target = target
        elif memory_regime == "manual":
            try:
                target = left_hw_target
                right_operand = _hw_target_handler(right_operand, target, right_type)
                right_hw_target = target
            except:
                raise ValueError(
                    f"Out of memory, try using automatic memory management."
                )
        else:
            raise ValueError(
                f"Invalid memory regime '{memory_regime}'. Supported regimes are {regime_list}."
            )
            # This should never happen here

    # Dispatch based on operation
    dispatch_func = _OPERATION_MAP[operation]
    return dispatch_func(left_operand, right_operand, left_type, right_type)


def _hw_target_handler(data, hw_target, matrix_type):
    # Moves data to hw_target
    if hw_target == "accelerator":
        if matrix_type == "sparse":
            return cu_sp.csr_matrix(data)
        if matrix_type == "dense":
            return cp.asarray(data)
    if hw_target == "host":
        if matrix_type == "sparse":
            return data.get()
        if matrix_type == "dense":
            return data.get()
    raise TypeError(f"Unknown hw_target type: {hw_target}")


def _get_dispatch_metadata(data) -> tuple[str, str]:
    """Determine metadata related to the data dispatch.

    Parameters
    ----------
    data : Union[np.ndarray, sp.spmatrix, cp.ndarray, cupyx.scipy.sparse.spmatrix]
        The input data for which to determine the dispatch metadata.

    Returns
    -------
    tuple[str, str]
        A tuple containing:
        - The type of the data ('sparse' or 'dense').
        - The hardware target of the data ('host' or 'accelerator').
    """
    # Basic type checking for host (CPU) data
    if sp.issparse(data):
        return "sparse", "host"
    if isinstance(data, np.ndarray):
        return "dense", "host"
    if isinstance(data, float):
        return "dense", "host"
    if isinstance(data, int):
        return "dense", "host"

    # Adding GPU types if cupy is available
    if cupy_version is not None:
        if cu_sp.issparse(data):
            return "sparse", "accelerator"
        if isinstance(data, cp.ndarray):
            return "dense", "accelerator"

    raise TypeError(f"Unknown matrix type: {type(data)}")


def _target_decider(
    left_data, right_data, left_hw_target, right_hw_target, left_type, right_type
):
    """Decide the hardware target for 'auto' memory regime based on available memory and data size"""
    gpus = GPUtil.getGPUs()

    used_memory = gpus[0].memoryUsed * 1024 * 1024  # Convert from MB to bytes
    total_memory = gpus[0].memoryTotal * 1024 * 1024  # Convert from MB to bytes
    available_memory = total_memory * memory_threshold - used_memory

    if left_hw_target == "host":
        if left_type == "sparse":
            if left_data.format == "coo":
                memory_needed = (
                    left_data.data.nbytes + left_data.row.nbytes + left_data.col.nbytes
                )
            else:
                memory_needed = (
                    left_data.data.nbytes
                    + left_data.indptr.nbytes
                    + left_data.indices.nbytes
                )
        else:
            memory_needed = left_data.nbytes

    if right_hw_target == "host":
        if right_type == "sparse":
            if right_data.format == "coo":
                memory_needed = (
                    right_data.data.nbytes
                    + right_data.row.nbytes
                    + right_data.col.nbytes
                )
            else:
                memory_needed = (
                    right_data.data.nbytes
                    + right_data.indptr.nbytes
                    + right_data.indices.nbytes
                )
        else:
            memory_needed = right_data.nbytes

    if memory_needed > available_memory:
        return "host"
    return "accelerator"
