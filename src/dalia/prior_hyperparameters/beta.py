# Copyright 2024-2025 DALIA authors. All rights reserved.
import scipy.stats as stats

from dalia import sp, xp
from dalia.configs.priorhyperparameters_config import (
    GaussianPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters
from dalia.utils.link_functions import scaled_logit


class BetaPriorHyperparameters(PriorHyperparameters):
    """Gaussian prior hyperparameters."""

    def __init__(
        self,
        config: GaussianPriorHyperparametersConfig,
    ) -> None:
        """Initializes the Gaussian prior hyperparameters."""
        super().__init__(config)

        self.alpha: float = config.alpha
        self.beta: float = config.beta


    def rescale_hyperparameters_to_internal(self, theta, direction):

        ### TODO: on longer term make scaled_logit default but let it be configurable in config
        ## beta prior is defined on [0,1], while BFGS works on (-inf, inf)
        if direction == "forward":
            theta_scaled = scaled_logit(theta, direction="forward")
        elif direction == "backward":
            theta_scaled = scaled_logit(theta, direction="backward")
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        if theta < 0 or theta > 1:
            ValueError(
                "Beta distribution is defined on the interval [0, 1]. theta: {theta}"
            )

        log_beta = (
            sp.special.gammaln(self.alpha)
            + sp.special.gammaln(self.beta)
            - sp.special.gammaln(self.alpha + self.beta)
        )
        log_prior = (
            (self.alpha - 1) * xp.log(theta)
            + (self.beta - 1) * xp.log(1 - theta)
            - log_beta
        )

        return log_prior
