# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np

from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.core.pyinla_config import PyinlaConfig


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
            self.mean_theta_spatio_temporal_variation = (
                pyinla_config.prior_hyperparameters.mean_theta_spatio_temporal_variation
            )

            self.precision_theta_spatial_range = (
                pyinla_config.prior_hyperparameters.precision_theta_spatial_range
            )
            self.precision_theta_temporal_range = (
                pyinla_config.prior_hyperparameters.precision_theta_temporal_range
            )
            self.precision_theta_spatio_temporal_variation = (
                pyinla_config.prior_hyperparameters.precision_theta_spatio_temporal_variation
            )

        if pyinla_config.likelihood.type == "gaussian":
            self.mean_theta_observations = (
                pyinla_config.prior_hyperparameters.mean_theta_observations
            )
            self.precision_theta_observations = (
                pyinla_config.prior_hyperparameters.precision_theta_observations
            )

    def evaluate_log_prior(
        self, theta_model: dict, theta_likelihood: dict
    ) -> np.ndarray:
        """Evaluate the log prior hyperparameters."""
        log_prior: float = 0.0

        # --- Log prior from model ---------------------------------------------
        if self.pyinla_config.model.type == "regression":
            pass
        elif self.pyinla_config.model.type == "spatio-temporal":
            log_prior_spatial_range = (
                self.precision_theta_spatial_range
                * (theta_model["spatial_range"] - self.mean_theta_spatial_range) ** 2
            )

            log_prior_temporal_range = (
                self.precision_theta_temporal_range
                * (theta_model["temporal_range"] - self.mean_theta_temporal_range) ** 2
            )

            log_prior_sd_spatio_temporal = (
                self.precision_theta_spatio_temporal_variation
                * (
                    theta_model["spatio_temporal_variation"]
                    - self.mean_theta_spatio_temporal_variation
                )
                ** 2
            )

            # print("theta_model['spatial_range']: ", theta_model["spatial_range"], ". mean_theta_spatial_range: ", self.mean_theta_spatial_range)
            # print("theta_model['temporal_range']: ", theta_model["temporal_range"], ". mean_theta_temporal_range: ", self.mean_theta_temporal_range)
            # print("theta_model['spatio_temporal_variation']: ", theta_model["spatio_temporal_variation"], ". mean_theta_spatio_temporal_variation: ", self.mean_theta_spatio_temporal_variation)

            log_prior += -0.5 * (
                log_prior_spatial_range
                + log_prior_temporal_range
                + log_prior_sd_spatio_temporal
            )

        # --- Log prior from likelihood ----------------------------------------
        if self.pyinla_config.likelihood.type == "gaussian":
            log_prior_likelihood = (
                self.precision_theta_observations
                * (
                    theta_likelihood["theta_observations"]
                    - self.mean_theta_observations
                )
                ** 2
            )

            log_prior += -0.5 * log_prior_likelihood
        elif self.pyinla_config.likelihood.type == "poisson":
            pass
        elif self.pyinla_config.likelihood.type == "binomial":
            pass

        return log_prior
