# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np

from pyinla import ArrayLike
from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.core.pyinla_config import PriorHyperparametersConfig


class GaussianPriorHyperparameters(PriorHyperparameters):
    """Gaussian prior hyperparameters."""

    def __init__(
        self,
        ph_config: PriorHyperparametersConfig,
        hyperparameter_type: str,
        **kwargs,
    ) -> None:
        """Initializes the Gaussian prior hyperparameters."""
        super().__init__(ph_config, hyperparameter_type)

        self.mean = ph_config.mean
        self.precision = ph_config.precision

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        return -0.5 * self.precision * (theta - self.mean) ** 2
