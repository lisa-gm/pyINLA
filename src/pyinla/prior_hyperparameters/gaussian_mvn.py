# Copyright 2024-2025 pyINLA authors. All rights reserved.
from pyinla import NDArray
from scipy.sparse import spmatrix

import numpy as np
from pyinla import sp, xp

from pyinla.configs.priorhyperparameters_config import (
    GaussianMVNPriorHyperparametersConfig,
)
from pyinla.core.prior_hyperparameters import PriorHyperparameters


class GaussianMVNPriorHyperparameters(PriorHyperparameters):
    """Gaussian MVN prior hyperparameters."""

    def __init__(
        self,
        config: GaussianMVNPriorHyperparametersConfig,
    ) -> None:
        """Initializes the Gaussian MVN prior hyperparameters."""
        super().__init__(config)

        self.mean: NDArray = config.mean
        self.precision: spmatrix = config.precision

        if xp == np:
            self.mean: NDArray = self.mean
            self.precision: spmatrix = self.precision
        else:
            self.mean: NDArray = xp.asarray(self.mean)
            self.precision: sp.sparse.spmatrix = sp.sparse.csc_matrix(self.precision)

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        # TODO: add check in config or somewhere else that dim(theta) and dim(mean) match
        if self.mean.shape != theta.shape:
            raise ValueError(
                f"Shape of theta ({theta.shape}) and mean ({self.mean.shape}) do not match."
            )

        if isinstance(self.mean, float):
            return -0.5 * self.precision * (theta - self.mean) ** 2
        else:
            # neglect constant as the precision is fixed
            return -0.5 * (theta - self.mean).T @ self.precision @ (theta - self.mean)
