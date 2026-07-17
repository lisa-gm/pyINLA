import pytest
import numpy as np
from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas import gemm, xxrk, trmm

from .conftest import INTERNAL_DEVICE_TYPES, DATA_TYPES