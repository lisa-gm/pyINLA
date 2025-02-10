# Copyright 2024-2025 pyINLA authors. All rights reserved.
from pyinla import NDArray
from scipy.sparse import spmatrix

from pyinla.configs.priorhyperparameters_config import (
    GaussianPriorHyperparametersConfig,
)
from pyinla.core.prior_hyperparameters import PriorHyperparameters


# class GaussianPriorHyperparameters(PriorHyperparameters):
#     """Gaussian prior hyperparameters."""

#     def __init__(
#         self,
#         config: GaussianPriorHyperparametersConfig,
#     ) -> None:
#         """Initializes the Gaussian prior hyperparameters."""
#         super().__init__(config)

#         self.mean: float = config.mean
#         self.precision: float = config.precision

#     def evaluate_log_prior(self, theta: float, **kwargs) -> float:
#         """Evaluate the log prior hyperparameters."""

#         return -0.5 * self.precision * (theta - self.mean) ** 2

class GaussianPriorHyperparameters(PriorHyperparameters):
    """Gaussian prior hyperparameters."""

    def __init__(
        self,
        config: GaussianPriorHyperparametersConfig,
    ) -> None:
        """Initializes the Gaussian prior hyperparameters."""
        super().__init__(config)

        self.mean: NDArray = config.mean
        self.precision: spmatrix = config.precision

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        # TODO: add check in config or somewhere else that dim(theta) and dim(mean) match

        # neglect constant as the precision is fixed
        return -0.5 * (theta - self.mean).T @ self.precision @ (theta - self.mean)
