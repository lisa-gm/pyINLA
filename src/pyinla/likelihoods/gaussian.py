# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import eye, sparray

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
        theta_likelihood: dict = None,
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
        if theta_likelihood is None:
            raise ValueError(
                "theta_likelihood must be provided to evaluate gaussian likelihood."
            )

        theta_observations = theta_likelihood["theta_observations"]

        yAx = y - a @ x

        likelihood = (
            0.5 * theta_observations * self.n_observations
            - 0.5 * yAx.T @ yAx * np.exp(theta_observations)
        )

        return likelihood

    def evaluate_gradient_likelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        """Evaluate the gradient of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        y : ArrayLike
            Vector of the observations.
        eta : ArrayLike
            Vector of the linear predictor.
        kwargs : dict
            Extra arguments.
            theta_likelihood : float
                Specific parameter for the likelihood calculation.

        Returns
        -------
        gradient_likelihood : ArrayLike
            Gradient of the likelihood.
        """
        if theta_likelihood is None:
            raise ValueError(
                "theta_likelihood must be provided to evaluate gaussian likelihood."
            )

        theta_observations = theta_likelihood["theta_observations"]

        gradient_likelihood = -np.exp(theta_observations) * (y - eta)

        return gradient_likelihood

    def evaluate_hessian_likelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        """Evaluate the Hessian of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        y : ArrayLike
            Vector of the observations.
        eta : ArrayLike
            Vector of the linear predictor.
        kwargs : dict
            Extra arguments.
            theta_likelihood : float
                Specific parameter for the likelihood calculation.

        Returns
        -------
        hessian_likelihood : ArrayLike
            Hessian of the likelihood.
        """
        if theta_likelihood is None:
            raise ValueError(
                "theta_likelihood must be provided to evaluate gaussian likelihood."
            )

        theta_observations = theta_likelihood["theta_observations"]

        hessian_likelihood = -np.exp(theta_observations) * eye(self.n_observations)

        return hessian_likelihood
