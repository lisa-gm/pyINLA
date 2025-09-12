# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia.communicator.communicator_config import DALIACommunicatorConfig
from dalia.communicator.communicator import Communicator


class DALIACommunicator:
    """
    DALIA-specific communicator class extending the base Communicator.

    This class manages multiple communicators, nested one into another, following
    the parallel function evaluation, precision matrix evaluation, and solver
    parallelization strategy.

    Attributes:
    -----------
    world : Communicator
        The global communicator for all processes.
    feval : Communicator
        The communicator at the function evaluation level.
    qeval : Communicator
        The communicator at the precision matrix evaluation level.
    solver : Communicator
        The communicator at the solver level.
    """

    def __init__(self, config: DALIACommunicatorConfig) -> None:
        self.config = config

        self.world = Communicator(config=config.base)
        self.feval = Communicator(config=config.feval)
        self.qeval = Communicator(config=config.qeval)
        self.solver = Communicator(config=config.solver)
