# src/backend/datastructures/matrix/core/__init__.py
# Internal API for matrix module
from .matrix import Matrix
from .dense import DenseMatrix
from .sparse import SparseMatrix

__all__ = ["Matrix", "DenseMatrix", "SparseMatrix"]
