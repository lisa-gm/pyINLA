# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from cupyx.profiler import time_range
from numpy.typing import ArrayLike
from scipy.sparse import eye

from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_observations: int,
    ) -> None:
        """Initializes the Gaussian likelihood."""
        super().__init__(pyinla_config, n_observations)

        self.theta = {"theta_observations": pyinla_config.likelihood.theta_observations}

    def get_theta(self) -> dict:
        """Get the likelihood initial hyperparameters."""
        return self.theta

    @time_range()
    def evaluate_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
    ) -> float:
        """Evaluate a Gaussian likelihood.

        Notes
        -----

        Evaluate Gaussian log-likelihood for a given set of observations, latent parameters, and design matrix, where
        the observations are assumed to be identically and independently distributed given eta (=A*x). Leading to:
        log (p(y|eta)) = -0.5 * n * log(2 * pi) - 0.5 * n * theta_observations - 0.5 * exp(theta_observations) * (y - eta)^T * (y - eta)
        where the constant in front of the likelihood is omitted.

        Parameters
        ----------
        eta : ArrayLike
            Vector of the linear predictor.
        y : ArrayLike
            Vector of the observations.
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

        yEta = eta - y

        likelihood = (
            0.5 * theta_observations * self.n_observations
            - 0.5 * np.exp(theta_observations) * yEta.T @ yEta
        )

        return likelihood

    @time_range()
    def evaluate_gradient_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        """Evaluate the gradient of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : ArrayLike
            Vector of the linear predictor.
        y : ArrayLike
            Vector of the observations.
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
        gradient_likelihood = -np.exp(theta_observations) * (eta - y)

        return gradient_likelihood

    @time_range()
    def evaluate_hessian_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        """Evaluate the Hessian of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : ArrayLike
            Vector of the linear predictor.
        y : ArrayLike
            Vector of the observations.
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
