# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np

from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig


class xxxx(Likelihood):
    """Gaussian prior hyperparameters."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        theta: np.ndarray,
        **kwargs,
    ) -> None:
        """Initializes the Gaussian prior hyperparameters."""
        super().__init__(pyinla_config)

        # Check if mean vector is provided in the input folder
        try:
            self.mean = np.load(pyinla_config.input_dir / "mean_hyperparameters.npy")
        except FileNotFoundError:
            self.mean = np.zeros_like(theta)

        # Check if variance vector is provided in the input folder
        try:
            self.variance = np.load(
                pyinla_config.input_dir / "variance_hyperparameters.npy"
            )
        except FileNotFoundError:
            self.variance = np.ones_like(theta)

    def evaluate_log_prior(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate the prior hyperparameters."""

        # Compute the log prior
        log_prior = -0.5 * np.sum((theta - self.mean) ** 2 / self.variance)

        return log_prior
