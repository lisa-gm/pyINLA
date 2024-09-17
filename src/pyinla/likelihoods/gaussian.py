# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import sparray

from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_observations: int,
        **kwargs,
    ) -> None:
        """Initializes the Gaussian likelihood."""
        super().__init__(pyinla_config, n_observations)

        self.theta_initial = {
            "theta_observations": pyinla_config.likelihood.theta_observations
        }

    def get_theta_initial(self) -> dict:
        """Get the likelihood initial hyperparameters."""
        return self.theta_initial

    def evaluate_likelihood(
        self,
        y: ArrayLike,
        a: sparray,
        x: ArrayLike,
        **kwargs,
    ) -> float:
        """Evaluate a Gaussian likelihood.

        Parameters
        ----------
        y : ArrayLike
            Vector of the observations.
        a : sparray
            Design matrix.
        x : ArrayLike
            Vector of the latent parameters.
        kwargs : dict
            Extra arguments.
            theta_likelihood : float
                Specific parameter for the likelihood calculation.

        Returns
        -------
        likelihood : float
            Likelihood.
        """
        theta_observations = kwargs["theta_observations"]

        yAx = y - a @ x

        likelihood = (
            0.5 * theta_observations * self.n_observations
            - 0.5 * yAx.T @ yAx * np.exp(theta_observations)
        )

        return likelihood
