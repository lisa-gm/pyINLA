# Copyright 2024-2025 pyINLA authors. All rights reserved.
import scipy.stats as stats


from pyinla.configs.priorhyperparameters_config import (
    GaussianPriorHyperparametersConfig,
)
from pyinla.core.prior_hyperparameters import PriorHyperparameters


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

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """Evaluate the log prior hyperparameters."""

        if theta < 0 or theta > 1:
            ValueError(
                "Beta distribution is defined on the interval [0, 1]. theta: {theta}"
            )

        log_prior = stats.beta.logpdf(theta, self.alpha, self.beta)
        return log_prior
