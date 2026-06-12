# src/dalia/backend/datastructures/matrix/dispatch/sub.py


def dispatch_sub(left, right, left_type, right_type, hw_target):
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
