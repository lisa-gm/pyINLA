# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia.communicator.dalia_communicator import DALIACommunicator
from dalia.communicator.communicator import Communicator
from dalia.communicator.communicator_config import (
    DALIACommunicatorConfig,
    CommunicatorConfig,
    parse_dalia_communicator_config,
)


class TestDALIACommunicatorBasic:
    """Basic unit tests for the DALIACommunicator class."""

    def test_dalia_communicator_initialization(self):
        """Test initialization of DALIACommunicator with default configs."""
        basic_config: CommunicatorConfig = {}
        dalia_comm_config: DALIACommunicatorConfig = parse_dalia_communicator_config(
            config=basic_config,
        )

        dalia_comm = DALIACommunicator(config=dalia_comm_config)

        # Check that each communicator is initialized correctly
        assert isinstance(dalia_comm.world, Communicator)
        assert isinstance(dalia_comm.feval, Communicator)
        assert isinstance(dalia_comm.qeval, Communicator)
        assert isinstance(dalia_comm.solver, Communicator)


if __name__ == "__main__":
    """Simple python runner of all tests in this module."""
    testing_dalia = TestDALIACommunicatorBasic()
    testing_dalia.test_dalia_communicator_initialization()
