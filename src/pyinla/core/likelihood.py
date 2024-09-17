# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from pyinla.core.pyinla_config import PyinlaConfig


class Likelihood(ABC):
    """Abstract core class for likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the likelihood."""

        self.pyinla_config = pyinla_config

    @abstractmethod
    def evaluate_likelihood(self, theta_likelihood: dict) -> float:
        pass
