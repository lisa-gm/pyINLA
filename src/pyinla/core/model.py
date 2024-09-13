# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC

from scipy.sparse import load_npz

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

        # Load observation vector
        self.y = load_npz(pyinla_config.input_dir / "y.npz")
        self.a = load_npz(pyinla_config.input_dir / "a.npz")

        self._check_dimensions_observations()

    def _check_dimensions_observations(self) -> None:
        """Check the dimensions of the model."""
        assert self.y.shape[0] == self.a.shape[0], "Dimensions of y and a do not match."
