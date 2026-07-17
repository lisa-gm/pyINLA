# src/dalia/backend/datastructures/__init__.py - Public API for users
from dalia.backend.datastructures.matrix.core import (
    DenseMatrix,
    Matrix,
    SparseMatrix,
    BStructMatrix,
    Vector,
)

__all__ = ["Matrix", "DenseMatrix", "SparseMatrix", "BStructMatrix", "Vector"]
