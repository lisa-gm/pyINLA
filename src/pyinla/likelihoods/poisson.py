# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import sparray

from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig


class PoissonLikelihood(Likelihood):
    """Poisson likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_observations: int,
        **kwargs,
    ) -> None:
        """Initializes the Poisson likelihood."""
        super().__init__(pyinla_config, n_observations)

        # Load the extra coeficients for Poisson likelihood
        try:
            self.e = np.load(pyinla_config.input_dir / "e.npy")
        except FileNotFoundError:
            self.e = np.ones((n_observations), dtype=int)

    def get_theta_initial(self) -> dict:
        """Get the likelihood initial hyperparameters."""
        return {}

    def evaluate_likelihood(
        self,
        y: ArrayLike,
        a: sparray,
        x: ArrayLike,
        theta_likelihood: dict = None,
    ) -> float:
        Ax = a @ x
        likelihood = np.dot(Ax, y) - np.sum(self.e * np.exp(Ax))

        return likelihood
