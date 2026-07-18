# tests/backend/units/linalg/lapack/conftest.py

import numpy as np
import pytest

from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

# Type groups - reusable across all tests
INTERNAL_DEVICE_TYPES = ["host"]
DATA_TYPES = ["float32", "float64", "complex64", "complex128"]

INTERNAL_DEVICE_TYPES.append(
    pytest.param(
        "accelerator",
        marks=pytest.mark.skipif(
            cupy_version is None,
            reason="CuPy is not installed",
        ),
    )
)

