# src/dalia/backend/datastructures/__init__.py - Public API for users
from dalia.backend.datastructures.matrix.core import (
    BStructMatrix,
    DenseMatrix,
    Matrix,
    SparseMatrix,
    Vector,
)

__all__ = ["Matrix", "DenseMatrix", "SparseMatrix", "BStructMatrix", "Vector"]
