# src/dalia/backend/datastructures/matrix/core/__init__.py - Internal API for matrix module
from .dense import DenseMatrix
from .matrix import Matrix
from .sparse import SparseMatrix
from .block_structured import BStructMatrix

__all__ = ["Matrix", "DenseMatrix", "SparseMatrix", "BStructMatrix"]
