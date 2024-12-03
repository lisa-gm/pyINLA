# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from pyinla.core.pyinla_config import PriorHyperparametersConfig


class PriorHyperparameters(ABC):
    """Abstract core class for prior hyperparameters."""

    def __init__(
        self,
        ph_config: PriorHyperparametersConfig,
        hyperparameter_type: str,
    ) -> None:
        """Initializes the prior hyperparameters."""

        self.ph_config: PriorHyperparametersConfig = ph_config
        self.hyperparameter_type: str = hyperparameter_type

    @abstractmethod
    def evaluate_log_prior(self, theta: float) -> float:
        """Evaluate the log prior hyperparameters."""
        pass
