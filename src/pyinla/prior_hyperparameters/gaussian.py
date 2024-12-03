# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.core.pyinla_config import GaussianPriorHyperparametersConfig


class GaussianPriorHyperparameters(PriorHyperparameters):
    """Gaussian prior hyperparameters."""

    def __init__(
        self,
        ph_config: GaussianPriorHyperparametersConfig,
        hyperparameter_type: str,
        **kwargs,
    ) -> None:
        """Initializes the Gaussian prior hyperparameters."""
        super().__init__(ph_config, hyperparameter_type)

        self.mean: float = ph_config.mean
        self.precision: float = ph_config.precision

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        return -0.5 * self.precision * (theta - self.mean) ** 2
