# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import sp_array

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

    def evaluate_likelihood(
        self,
        y: ArrayLike,
        a: sp_array,
        x: ArrayLike,
        **kwargs,
    ) -> float:
        """Evaluate a Gaussian likelihood.

        Parameters
        ----------
        y : ArrayLike
            Vector of the observations.
        a : sp_array
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
        theta_likelihood = kwargs["theta_likelihood"]

        yAx = y - a @ x

        likelihood = (
            0.5 * theta_likelihood * self.n_observations
            - 0.5 * yAx.T @ yAx * np.exp(theta_likelihood)
        )

        return likelihood
