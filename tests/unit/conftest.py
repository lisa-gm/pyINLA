# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Global pytest fixtures for the Serinv tests.

import os
import pytest

from dalia import backend_flags

ARRAY_MODULE = [
    pytest.param("numpy", id="numpy"),
]
if backend_flags["cupy_avail"]:
    ARRAY_MODULE.append(
        [
            pytest.param("cupy", id="cupy"),
        ]
    )


@pytest.fixture(params=ARRAY_MODULE, autouse=True)
def ARRAY_MODULE(request: pytest.FixtureRequest) -> str:
    return request.param


COMM_LIB = [
    pytest.param("none", id="none"),
]
if backend_flags["mpi_avail"]:
    COMM_LIB.append(
        pytest.param("host_mpi", id="host_mpi"),
    )
    if backend_flags["mpi_cuda_aware"]:
        COMM_LIB.append(
            pytest.param("device_mpi", id="device_mpi"),
        )
    if backend_flags["nccl_avail"]:
        COMM_LIB.append(
            pytest.param("nccl", id="nccl"),
        )


@pytest.fixture(params=COMM_LIB, autouse=True)
def COMM_LIB(request: pytest.FixtureRequest) -> str:
    return request.param
