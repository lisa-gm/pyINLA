# src/backend/datastructures/__init__.py - Public API for users
from backend.datastructures.matrix.core import DenseMatrix, Matrix, SparseMatrix

__all__ = ["Matrix", "DenseMatrix", "SparseMatrix"]
