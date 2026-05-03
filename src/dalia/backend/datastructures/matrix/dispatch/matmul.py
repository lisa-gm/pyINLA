# src/dalia/backend/datastructures/matrix/dispatch/matmul.py
from .gemm import gemm

def dispatch_matmul(left, right, left_type, right_type, hw_target):
    """Dispatch matrix multiplication to optimized backends"""
    if left_type == "sparse" and right_type == "dense":
        return left @ right

    if left_type == "dense" and right_type == "sparse":
        return left @ right

    if left_type == "sparse" and right_type == "sparse":
        return left @ right

    if left_type == "dense" and right_type == "dense":
        return gemm(left, right, hw_target)

    raise TypeError(f"Cannot multiply {left_type} @ {right_type}")
