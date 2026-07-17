# src/dalia/backend/datastructures/matrix/dispatch/sub.py
from typing import Literal, Union

import numpy as np
import scipy.sparse as sp


def dispatch_sub(
    left: Union[np.ndarray, sp.spmatrix],
    right: Union[np.ndarray, sp.spmatrix],
    left_type: Literal["sparse", "dense"],
    right_type: Literal["sparse", "dense"],
) -> Union[np.ndarray, sp.spmatrix]:
    """Dispatch matrix subtraction to optimized backends"""

    if left_type == "sparse" and right_type == "dense":
        return left - right

    if left_type == "dense" and right_type == "sparse":
        return left - right

    if left_type == "sparse" and right_type == "sparse":
        return left - right

    if left_type == "dense" and right_type == "dense":
        return left - right

    raise TypeError(f"Cannot subtract {left_type} - {right_type}")
