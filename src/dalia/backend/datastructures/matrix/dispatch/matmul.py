# src/dalia/backend/datastructures/matrix/dispatch/matmul.py


def dispatch_matmul(left, right, left_type, right_type):
    """Dispatch matrix multiplication to optimized backends"""
    if left_type == "sparse" and right_type == "dense":
        return left @ right

    if left_type == "dense" and right_type == "sparse":
        return left @ right

    if left_type == "sparse" and right_type == "sparse":
        return left @ right

    if left_type == "dense" and right_type == "dense":
        return left @ right

    raise TypeError(f"Cannot multiply {left_type} @ {right_type}")
