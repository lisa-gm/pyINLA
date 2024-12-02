# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np

from pyinla import xp
from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.core.pyinla_config import PriorHyperparametersConfig


class PenalizedComplexityPriorHyperparameters(PriorHyperparameters):
    """Penalized Complexity prior hyperparameters."""

    def __init__(
        self,
        ph_config: PriorHyperparametersConfig,
        hyperparameter_type: str,
        **kwargs,
    ) -> None:
        """Initializes the Penalized Complexity prior hyperparameters."""
        super().__init__(ph_config, hyperparameter_type)

        self.alpha = ph_config.alpha
        self.u = ph_config.u

        self.lambda_theta = 0.0

        if self.hyperparameter_type == "r_s":
            spatial_dim = kwargs["spatial_dim", 2]

            self.lambda_theta = -xp.log(self.alpha) * pow(
                self.u,
                0.5 * spatial_dim,
            )
        elif self.hyperparameter_type == "r_t":
            self.lambda_theta = -xp.log(self.alpha) * pow(self.u, 0.5)
        elif self.hyperparameter_type == "sigma_st":
            self.lambda_theta = -xp.log(self.alpha) / self.u
        elif self.hyperparameter_type == "prec_o":
            self.lambda_theta = -xp.log(self.alpha) / self.u

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the prior hyperparameters."""
        log_prior = 0.0

        if self.hyperparameter_type == "r_s":
            spatial_dim = kwargs["spatial_dim", 2]

            if spatial_dim == 2:
                log_prior = (
                    xp.log(self.lambda_theta)
                    - self.lambda_theta * xp.exp(theta)
                    - theta
                )
            else:
                raise ValueError("Not implemented for other than 2D spatial domains")
        elif self.hyperparameter_type == "r_t":
            log_prior = (
                xp.log(self.lambda_theta)
                - self.lambda_theta * xp.exp(0.5 * theta)
                + xp.log(0.5)
                - theta
            )
        elif self.hyperparameter_type == "sigma_st":
            log_prior = (
                xp.log(self.lambda_theta) - self.lambda_theta * xp.exp(theta) + theta
            )
        elif self.hyperparameter_type == "prec_o":
            log_prior = (
                xp.log(self.lambda_theta) - self.lambda_theta * xp.exp(theta) - theta
            )

        return log_prior
