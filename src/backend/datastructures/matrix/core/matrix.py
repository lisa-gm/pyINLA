"""
Design choices: Explicit sub-class for Sparse, Dense and Structured cases.

A Matrix class is the base class of all matrix types. There is no automatic dispatch
and the user need to create the subclass that matches the correct matrix type.
We opted for this coice as the sparsity is know from a statistical model implementation perspective.

The operatios are dispatched automatically based on the type of the operands.


"""

from abc import ABC

from backend.datastructures.matrix.dispatch import blas_dispatch, Operation

from .utils import wrap_result


class Matrix(ABC):
    def __init__(self, data):
        self._data = data

    def __matmul__(self, other):
        other_data = other._data if isinstance(other, Matrix) else other
        result_data = blas_dispatch(Operation.MATMUL, self._data, other_data)
        return self._wrap_result(
            result_data
        )  # Wrapping to correct Matrix() happens here

    def _wrap_result(self, data):
        """Wrap the result data in the appropriate Matrix subclass"""
        return wrap_result(data)
