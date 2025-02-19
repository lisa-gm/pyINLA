# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import xp
from pyinla.configs.priorhyperparameters_config import (
    PenalizedComplexityPriorHyperparametersConfig,
)
from pyinla.core.prior_hyperparameters import PriorHyperparameters


class PenalizedComplexityPriorHyperparameters(PriorHyperparameters):
    """Penalized Complexity prior hyperparameters."""

    def __init__(
        self,
        config: PenalizedComplexityPriorHyperparametersConfig,
        **kwargs,
    ) -> None:
        """Initializes the Penalized Complexity prior hyperparameters."""
        super().__init__(config)

        self.hyperparameter_type: str = kwargs.get("hyperparameter_type")

        self.alpha: float = config.alpha
        self.u: float = config.u

        self.lambda_theta: float = 0.0

        if self.hyperparameter_type == "r_s":
            spatial_dim: int = 2  # kwargs["spatial_dim", 2]

            self.lambda_theta = -xp.log(self.alpha) * pow(
                self.u,
                0.5 * spatial_dim,
            )
        elif self.hyperparameter_type == "r_t":
            self.lambda_theta = -xp.log(self.alpha) * pow(self.u, 0.5)
        elif (
            self.hyperparameter_type == "sigma_st"
            or self.hyperparameter_type == "sigma_e"
        ):
            self.lambda_theta = -xp.log(self.alpha) / self.u
        elif self.hyperparameter_type == "prec_o":
            self.lambda_theta = -xp.log(self.alpha) / self.u

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the prior hyperparameters."""
        log_prior: float = 0.0

        if self.hyperparameter_type == "r_s":
            spatial_dim: int = 2  # kwargs["spatial_dim", 2]

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
