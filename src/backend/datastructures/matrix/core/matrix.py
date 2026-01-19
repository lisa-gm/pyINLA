"""
Design choices: Explicit sub-class for Sparse, Dense and Structured cases.

A Matrix class is the base class of all matrix types. There is no automatic dispatch
and the user need to create the subclass that matches the correct matrix type.
We opted for this coice as the sparsity is know from a statistical model implementation perspective.

The operatios are dispatched automatically based on the type of the operands.


"""

import numpy as np
import scipy.sparse as sp

from backend.datastructures.matrix.dispatch import blas_dispatch, Operation


class Matrix:
    def __init__(self, data):
        self._data = data

    def __matmul__(self, other):
        other_data = other._data if isinstance(other, Matrix) else other
        result_data = blas_dispatch(Operation.MATMUL, self._data, other_data)
        return self._wrap_result(result_data)  # ← Wrapping happens here

    def _wrap_result(self, data):
        """Wrap the result data in the appropriate Matrix subclass"""
        if sp.issparse(data):
            return SparseMatrix(data)
        elif isinstance(data, np.ndarray):
            return DenseMatrix(data)
        elif isinstance(data, StructuredMatrix):
            return StructuredMatrix(data)
        else:
            raise TypeError(f"Unknown matrix type: {type(data)}")
