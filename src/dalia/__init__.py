# src/dalia/__init__.py

from dalia.__about__ import __version__

# from dalia.inla.core.dalia import DALIA
from dalia import statistical_modeling_toolbox

cupy_version = None
try:
    import cupy as cp
    cupy_version = cp.__version__
except ImportError:
    pass


__all__ = [
    "__version__",
    "DALIA",
    "statistical_modeling_toolbox",
    "cupy_version",
]
