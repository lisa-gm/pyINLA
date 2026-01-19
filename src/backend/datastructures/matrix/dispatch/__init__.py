# dispatch/__init__.py
from .dispatcher import blas_dispatch
from .operations import Operation

__all__ = ["blas_dispatch", "Operation"]

# Now users can:
# from backend.datastructures.matrix.dispatch import blas_dispatch, Operation
