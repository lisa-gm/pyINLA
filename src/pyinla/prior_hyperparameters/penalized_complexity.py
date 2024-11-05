# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np

from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.core.pyinla_config import PyinlaConfig


class PenalizedComplexityPriorHyperparameters(PriorHyperparameters):
    """Penalized Complexity prior hyperparameters."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        **kwargs,
    ) -> None:
        """Initializes the Penalized Complexity prior hyperparameters."""
        super().__init__(pyinla_config)

        if pyinla_config.model_type == "spatio-temporal":
            self.alpha_theta_spatial_range = (
                pyinla_config.prior_hyperparameters.alpha_theta_spatial_range
            )
            self.alpha_theta_temporal_range = (
                pyinla_config.prior_hyperparameters.alpha_theta_temporal_range
            )
            self.alpha_theta_spatio_temporal_variation = (
                pyinla_config.prior_hyperparameters.alpha_theta_spatio_temporal_variation
            )

            self.u_theta_spatial_range = (
                pyinla_config.prior_hyperparameters.u_theta_spatial_range
            )
            self.u_theta_temporal_range = (
                pyinla_config.prior_hyperparameters.u_theta_temporal_range
            )
            self.u_theta_spatio_temporal_variation = (
                pyinla_config.prior_hyperparameters.u_theta_spatio_temporal_variation
            )

            self.lambda_theta_spatial_range = -np.log(
                self.alpha_theta_spatial_range
            ) * pow(
                self.u_theta_spatial_range,
                0.5 * pyinla_config.model.spatial_domain_dimension,
            )
            self.lambda_theta_temporal_range = -np.log(
                self.alpha_theta_temporal_range
            ) * pow(self.u_theta_temporal_range, 0.5)
            self.lambda_theta_spatio_temporal_variation = (
                -np.log(self.alpha_theta_spatio_temporal_variation)
                / self.u_theta_spatio_temporal_variation
            )

        if pyinla_config.likelihood.type == "gaussian":
            self.theta_observations = pyinla_config.likelihood.theta_observations

            self.lambda_theta_observations = (
                -np.log(pyinla_config.prior_hyperparameters.alpha_theta_observations)
                / pyinla_config.prior_hyperparameters.u_theta_observations
            )

    # TODO: input theta in interpretable scale
    def evaluate_log_prior(
        self, theta_model: dict, theta_likelihood: dict
    ) -> np.ndarray:
        """Evaluate the prior hyperparameters."""
        log_prior = 0.0

        # --- Log prior from model ---------------------------------------------
        if self.pyinla_config.model.type == "regression":
            pass
        elif self.pyinla_config.model_type == "spatio-temporal":
            if self.pyinla_config.model.spatial_domain_dimension == 2:
                # 2-D... should be correct
                log_prior += (
                    np.log(self.lambda_theta_spatial_range)
                    - self.lambda_theta_spatial_range
                    * np.exp(theta_model["spatial_range"])
                    - theta_model["spatial_range"]
                )

                log_prior += (
                    np.log(self.lambda_theta_temporal_range)
                    - self.lambda_theta_temporal_range
                    * np.exp(0.5 * theta_model["temporal_range"])
                    + np.log(0.5)
                    - theta_model["temporal_range"]
                )

                log_prior += (
                    np.log(self.lambda_theta_spatio_temporal_variation)
                    - self.lambda_theta_spatio_temporal_variation
                    * np.exp(theta_model["spatio_temporal_variation"])
                    + theta_model["spatio_temporal_variation"]
                )
            else:
                raise ValueError("Not implemented for other than 2D cases")

        # --- Log prior from likelihood ----------------------------------------
        if self.pyinla_config.likelihood.type == "gaussian":
            log_prior += (
                np.log(self.lambda_theta_observations)
                - self.lambda_theta_observations
                * np.exp(theta_likelihood["observations"])
                - theta_likelihood["observations"]
            )
        elif self.pyinla_config.likelihood.type == "poisson":
            pass
        elif self.pyinla_config.likelihood.type == "binomial":
            pass

        return log_prior
