# Copyright 2024-2025 pyINLA authors. All rights reserved.
from pyinla import NDArray
from scipy.sparse import spmatrix

import numpy as np

from pyinla.configs.priorhyperparameters_config import (
    GaussianMVNPriorHyperparametersConfig,
)
from pyinla.core.prior_hyperparameters import PriorHyperparameters


class GaussianMVNPriorHyperparameters(PriorHyperparameters):
    """Gaussian prior hyperparameters."""

    def __init__(
        self,
        config: GaussianMVNPriorHyperparametersConfig,
    ) -> None:
        """Initializes the Gaussian prior hyperparameters."""
        super().__init__(config)

        self.mean: NDArray = config.mean
        self.precision: spmatrix = config.precision

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        # TODO: add check in config or somewhere else that dim(theta) and dim(mean) match

        if isinstance(self.mean, float):
            return -0.5 * self.precision * (theta - self.mean) ** 2

        else:
            # neglect constant as the precision is fixed
            return -0.5 * (theta - self.mean).T @ self.precision @ (theta - self.mean)


if __name__ == "__main__":

    # Test GaussianPriorHyperparameters
    config = GaussianMVNPriorHyperparametersConfig(mean=0, precision=2)
    prior_hyperparameters = GaussianMVNPriorHyperparameters(config)

    theta = 1
    log_prior = prior_hyperparameters.evaluate_log_prior(theta)
    print(log_prior)  # -0.0

    n = 3
    mean = np.ones((n, 1))
    precision = 2 * np.eye(n)

    config = GaussianMVNPriorHyperparametersConfig(mean=mean, precision=precision)

    theta = np.random.randn(n, 1)
    prior_hyperparameters = GaussianMVNPriorHyperparameters(config)
    log_prior = prior_hyperparameters.evaluate_log_prior(theta)
    print(log_prior)  # -0.0
