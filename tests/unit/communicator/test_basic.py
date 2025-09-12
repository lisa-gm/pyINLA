# Copyright 2024-2025 DALIA authors. All rights reserved.

import pytest

from dalia.communicator.communicator import Communicator
from dalia.communicator.communicator_config import (
    CommunicatorConfig,
    parse_dalia_communicator_config,
)


class TestCommunicatorBasic:
    """Basic unit tests for the Communicator class."""

    def test_communicator_initialization(self):
        """Test initialization of DALIACommunicator with default configs."""
        comm_config: CommunicatorConfig = CommunicatorConfig()
        communicator: Communicator = Communicator(config=comm_config)

        # Check that effective_comm_lib is resolved
        assert communicator._general_comm_lib in [
            "nccl",
            "device_mpi",
            "host_mpi",
            "none",
        ]
        assert communicator.rank is not None
        assert communicator.size is not None
        assert hasattr(communicator, "_collective_config")

        # Test that all collective configs are valid
        for attr in dir(communicator._collective_config):
            if not attr.startswith("_"):
                value = getattr(communicator._collective_config, attr)
                assert value in [
                    "default",
                    "nccl",
                    "device_mpi",
                    "host_mpi",
                    "none",
                ]

    def test_basic_functions(self):
        """Test basic functions of the Communicator."""
        comm_config: CommunicatorConfig = CommunicatorConfig()
        communicator: Communicator = Communicator(config=comm_config)

        # Test barrier
        communicator.barrier()


if __name__ == "__main__":
    """Simple python runner of all tests in this module."""
    testing_comm = TestCommunicatorBasic()
    testing_comm.test_communicator_initialization()
    testing_comm.test_basic_functions()
