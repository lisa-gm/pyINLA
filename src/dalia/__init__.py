# src/dalia/__init__.py

from dalia.__about__ import __version__

# from dalia.inla.core.dalia import DALIA
from dalia import statistical_modeling_toolbox
from dalia.backend.config import set_default_hw_target

cupy_version = None
target_list = ["host"]
try:
    import cupy
    cupy_version = cupy.__version__
    target_list.append("accelerator")
    set_default_hw_target("accelerator")
except ImportError:
    set_default_hw_target("host")
    pass
    

__all__ = [
    "__version__",
    "DALIA",
    "statistical_modeling_toolbox",
    "cupy_version",
    "target_list",
]