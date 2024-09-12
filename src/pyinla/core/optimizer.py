# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from pyinla.core.pyinla_config import PyinlaConfig


class Optimizer(ABC):
    """Abstract core class for optimizers."""

    @property
    @abstractmethod
    def system(self) -> str:
        ...

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the optimizer."""
        pass
