# Copyright 2024-2025 DALIA authors. All rights reserved.
from dalia import NDArray
from scipy.sparse import spmatrix
from dalia import sp, xp

import numpy as np

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

    def rescale_hyperparameters_to_internal(self, theta, direction):
        
        if direction == "forward":
            theta_scaled = xp.log(theta)
        elif direction == "backward":
            theta_scaled = xp.exp(theta)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        return -0.5 * self.precision * (theta - self.mean) ** 2
