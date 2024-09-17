# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC

from pyinla.core.pyinla_config import PyinlaConfig


class Model(ABC):
    """Abstract core class for statistical models."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the model."""
        self.pyinla_config = pyinla_config

        self.nb = self.pyinla_config.model.n_fixed_effects
