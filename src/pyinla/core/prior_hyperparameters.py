# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from pyinla.core.pyinla_config import PyinlaConfig


class PriorHyperparameters(ABC):
    """Abstract core class for prior hyperparameters."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the prior hyperparameters."""

        self.pyinla_config = pyinla_config
