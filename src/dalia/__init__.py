# src/dalia/__init__.py

from dalia.__about__ import __version__

# from dalia.inla.core.dalia import DALIA
from dalia import statistical_modeling_toolbox
from dalia.backend.config import check_cupy_availability, set_default_hw_target

if check_cupy_availability() is not None:
    set_default_hw_target("accelerator")
    

__all__ = [
    "__version__",
    "DALIA",
    "statistical_modeling_toolbox",
]


