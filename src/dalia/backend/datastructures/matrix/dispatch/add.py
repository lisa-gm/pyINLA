# src/dalia/backend/datastructures/matrix/dispatch/add.py


def dispatch_add(left, right, left_type, right_type, hw_target):
    """Dispatch matrix addition to optimized backends"""
    if left_type == "sparse" and right_type == "dense":
        return left + right

    if left_type == "dense" and right_type == "sparse":
        return left + right

    if left_type == "sparse" and right_type == "sparse":
        return left + right

    if left_type == "dense" and right_type == "dense":
        return left + right

    raise TypeError(f"Cannot add {left_type} + {right_type}")
