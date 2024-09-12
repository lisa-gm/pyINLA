# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla.core.model import Model
from pyinla.core.pyinla_config import PyinlaConfig


class SpatioTemporal(Model):
    """Fit a spatio-temporal model."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        super().__init__(pyinla_config)
