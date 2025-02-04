# Copyright 2024-2025 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from pyinla.configs.priorhyperparameters_config import PriorHyperparametersConfig


class PriorHyperparameters(ABC):
    """Abstract core class for prior hyperparameters."""

    def __init__(
        self,
        config: PriorHyperparametersConfig,
    ) -> None:
        """Initializes the prior hyperparameters."""

        self.config: PriorHyperparametersConfig = config

    @abstractmethod
    def evaluate_log_prior(self, theta: float) -> float:
        """Evaluate the log prior hyperparameters."""
        pass
