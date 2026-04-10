# src/dalia/__init__.py

from dalia.__about__ import __version__

# from dalia.inla.core.dalia import DALIA
from dalia import statistical_modeling_toolbox

__all__ = [
    "__version__",
    "DALIA",
    "statistical_modeling_toolbox",
]
