# tests/backend/units/datastructures/matrix/conftest.py

import pytest

from dalia.backend.config import cupy_version

# Type groups - reusable across all tests
EXTERNAL_SPARSE_TYPES = ["scipy_csr", "scipy_csc", "scipy_coo"]
EXTERNAL_DENSE_TYPES = ["numpy"]
INTERNAL_DEVICE_TYPES = ["host"]
MEMORY_REGIMES = ["manual"]


EXTERNAL_DENSE_TYPES.append(pytest.param("cupy", marks=pytest.mark.skipif(
                cupy_version is None,
                reason="CuPy is not installed",
            ),))
EXTERNAL_SPARSE_TYPES.append(pytest.param("cupy_csr", marks=pytest.mark.skipif(
                cupy_version is None,
                reason="CuPy is not installed",
            ),))
EXTERNAL_SPARSE_TYPES.append(pytest.param("cupy_csc", marks=pytest.mark.skipif(
                cupy_version is None,
                reason="CuPy is not installed",
            ),))
EXTERNAL_SPARSE_TYPES.append(pytest.param("cupy_coo", marks=pytest.mark.skipif(
                cupy_version is None,
                reason="CuPy is not installed",
            ),))
INTERNAL_DEVICE_TYPES.append(pytest.param("accelerator", marks=pytest.mark.skipif(
                cupy_version is None,
                reason="CuPy is not installed",
            ),))
MEMORY_REGIMES.append(pytest.param("auto", marks=pytest.mark.skipif(
                True,
                reason="No reason to test",
            ),))


