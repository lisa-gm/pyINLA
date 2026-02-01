# dispatch/__init__.py
from .dispatcher import blas_dispatch
from .operations import Operation

__all__ = ["blas_dispatch", "Operation"]
