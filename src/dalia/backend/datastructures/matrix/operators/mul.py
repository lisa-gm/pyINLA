# src/dalia/backend/datastructures/matrix/dispatch/mul.py
from typing import Literal, Union

import numpy as np
import scipy.sparse as sp


def dispatch_mul(
    left: Union[np.ndarray, sp.spmatrix],
    right: Union[np.ndarray, sp.spmatrix],
    left_type: Literal["sparse", "dense"],
    right_type: Literal["sparse", "dense"],
) -> Union[np.ndarray, sp.spmatrix]:
    """Dispatch matrix multiplication to optimized backends"""

    if left_type == "sparse" and right_type == "dense":
        return left * right

    if left_type == "dense" and right_type == "sparse":
        return left * right

    if left_type == "sparse" and right_type == "sparse":
        return left * right

    if left_type == "dense" and right_type == "dense":
        return left * right

    raise TypeError(f"Cannot multiply {left_type} * {right_type}")
