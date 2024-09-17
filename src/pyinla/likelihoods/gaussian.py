# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        **kwargs,
    ) -> None:
        """Initializes the Gaussian likelihood."""
        super().__init__(pyinla_config)

    def evaluate_likelihood(self, theta_likelihood: dict) -> float:
        pass
