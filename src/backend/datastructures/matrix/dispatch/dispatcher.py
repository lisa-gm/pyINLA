from .operations import Operation


def blas_dispatch(operation: Operation, left, right):
    # Type checking
    left_type = _get_matrix_type(left)
    right_type = _get_matrix_type(right)

    # Dispatch based on operation
    if operation == Operation.MATMUL:
        return _dispatch_matmul(left, right, left_type, right_type)
    elif operation == Operation.ADD:
        return _dispatch_add(left, right, left_type, right_type)
    # ...
