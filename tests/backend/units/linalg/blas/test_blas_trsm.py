import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from dalia.backend.blas import gemm, trmm, xxrk

from .conftest import DATA_TYPES, INTERNAL_DEVICE_TYPES
