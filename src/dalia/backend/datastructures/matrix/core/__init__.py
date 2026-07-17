# src/dalia/backend/datastructures/matrix/core/__init__.py - Internal API for matrix module
from .block_structured import BStructMatrix
from .dense import DenseMatrix
from .matrix import Matrix
from .sparse import SparseMatrix
from .vector_dense import Vector

__all__ = ["Matrix", "DenseMatrix", "SparseMatrix", "BStructMatrix", "Vector"]
