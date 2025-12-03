# Copyright 2024-2025 DALIA authors. All rights reserved.
import numpy as np
from scipy.sparse import spmatrix

from dalia import NDArray
from dalia.configs.priorhyperparameters_config import (
    GaussianPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class GaussianPriorHyperparameters(PriorHyperparameters):
    """Gaussian prior hyperparameters."""

    def __init__(
        self,
        config: GaussianPriorHyperparametersConfig,
    ) -> None:
        """Initializes the Gaussian prior hyperparameters."""
        super().__init__(config)

        self.mean: float = config.mean
        self.precision: float = config.precision

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        return -0.5 * self.precision * (theta - self.mean) ** 2
