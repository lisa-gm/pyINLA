# tests/backend/units/linalg/solvers/conftest.py

import pytest

from dalia.backend.config import cupy_version

# Type groups - reusable across all tests
INTERNAL_DEVICE_TYPES = ["host"]
INTERNAL_MATRIX_TYPES = ["SparseMatrix", "DenseMatrix"]

INTERNAL_DEVICE_TYPES.append(
    pytest.param(
        "accelerator",
        marks=pytest.mark.skipif(
            cupy_version is None,
            reason="CuPy is not installed",
        ),
    )
)
