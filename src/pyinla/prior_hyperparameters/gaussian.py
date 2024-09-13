# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np

from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.utils.theta_utils import make_theta_array


class GaussianPriorHyperparameters(PriorHyperparameters):
    """Gaussian prior hyperparameters."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        **kwargs,
    ) -> None:
        """Initializes the Gaussian prior hyperparameters."""
        super().__init__(pyinla_config)

        if pyinla_config.model.type == "spatio-temporal":
            self.mean_theta_spatial_range = (
                pyinla_config.prior_hyperparameters.mean_theta_spatial_range
            )
            self.mean_theta_temporal_range = (
                pyinla_config.prior_hyperparameters.mean_theta_temporal_range
            )
            self.mean_theta_sd_spatio_temporal = (
                pyinla_config.prior_hyperparameters.mean_theta_sd_spatio_temporal
            )

            self.variance_theta_spatial_range = (
                pyinla_config.prior_hyperparameters.variance_theta_spatial_range
            )
            self.variance_theta_temporal_range = (
                pyinla_config.prior_hyperparameters.variance_theta_temporal_range
            )
            self.variance_theta_sd_spatio_temporal = (
                pyinla_config.prior_hyperparameters.variance_theta_sd_spatio_temporal
            )

    def evaluate_log_prior(
        self, theta_model: dict, theta_likelihood: dict
    ) -> np.ndarray:
        """Evaluate the log prior hyperparameters."""
        theta = make_theta_array(theta_model, theta_likelihood)
        # TODO: Fix the mean and variance
        log_prior = -0.5 * np.sum((theta - self.mean) ** 2 / self.variance)

        return log_prior
