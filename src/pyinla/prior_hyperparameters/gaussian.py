# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla.configs.priorhyperparameters_config import (
    GaussianPriorHyperparametersConfig,
)
from pyinla.core.prior_hyperparameters import PriorHyperparameters


class GaussianPriorHyperparameters(PriorHyperparameters):
    """Gaussian prior hyperparameters."""

    def __init__(
        self,
        hyperparameter_type: str,
        config: GaussianPriorHyperparametersConfig,
        **kwargs,
    ) -> None:
        """Initializes the Gaussian prior hyperparameters."""
        super().__init__(config, hyperparameter_type)

        self.mean: float = config.mean
        self.precision: float = config.precision

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        return -0.5 * self.precision * (theta - self.mean) ** 2
