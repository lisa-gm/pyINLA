# Copyright 2024-2025 pyINLA authors. All rights reserved.
from pyinla import NDArray
from scipy.sparse import spmatrix

import numpy as np

from pyinla.configs.priorhyperparameters_config import (
    GaussianPriorHyperparametersConfig,
)
from pyinla.core.prior_hyperparameters import PriorHyperparameters


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
