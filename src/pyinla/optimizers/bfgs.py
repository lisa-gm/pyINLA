# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla.core.optimizer import Optimizer
from pyinla.core.pyinla_config import PyinlaConfig


class BFGS(Optimizer):
    """Fit the model using the BFGS algorithm."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        super().__init__(pyinla_config)
