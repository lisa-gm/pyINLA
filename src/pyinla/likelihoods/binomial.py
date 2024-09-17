# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np

from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig


class BinomialLikelihood(Likelihood):
    """Binomial likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        **kwargs,
    ) -> None:
        """Initializes the Binomial likelihood."""
        super().__init__(pyinla_config)

    def evaluate_likelihood(self, theta_likelihood: dict) -> float:
        pass
