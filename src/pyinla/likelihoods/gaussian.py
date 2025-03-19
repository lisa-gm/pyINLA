# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import ArrayLike, NDArray, sp, xp
from pyinla.configs.likelihood_config import GaussianLikelihoodConfig
from pyinla.core.likelihood import Likelihood


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood."""

    def __init__(
        self,
        n_observations: int,
        config: GaussianLikelihoodConfig,
    ) -> None:
        """Initializes the Gaussian likelihood."""
        super().__init__(n_observations, config)

    def evaluate_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
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
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        kwargs :
            theta : float
                Specific parameter for the likelihood calculation.

        Returns
        -------
        likelihood : float
            Likelihood.
        """

        theta: NDArray = kwargs.get("theta", None)
        if theta is None:
            raise ValueError("theta must be provided to evaluate gaussian likelihood.")

        yEta = eta - y
        # print("xp.exp(theta) in lh:", xp.exp(theta))

        likelihood: float = (
            0.5 * theta * self.n_observations - 0.5 * xp.exp(theta) * yEta.T @ yEta
        )

        return likelihood

    def evaluate_gradient_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        """Evaluate the gradient of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        kwargs :
            theta : float
                Specific parameter for the likelihood calculation.

        Returns
        -------
        gradient_likelihood : NDArray
            Gradient of the likelihood.
        """

        theta: NDArray = kwargs.get("theta", None)
        if theta is None:
            raise ValueError(
                "theta must be provided to evaluate gradient of gaussian likelihood."
            )

        gradient_likelihood: NDArray = -xp.exp(theta) * (eta - y)

        return gradient_likelihood

    def evaluate_hessian_likelihood(
        self,
        **kwargs,
    ) -> ArrayLike:
        """Evaluate the Hessian of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        kwargs :
            theta : float
                Specific parameter for the likelihood calculation.

        Returns
        -------
        hessian_likelihood : ArrayLike
            Hessian of the likelihood.
        """
        theta: float = kwargs.get("theta")
        if theta is None:
            raise ValueError(
                "theta must be provided to evaluate gradient of gaussian likelihood."
            )

        # print("hessian lh: xp.exp(theta)", xp.exp(theta))

        hessian_likelihood: ArrayLike = -xp.exp(theta) * sp.sparse.eye(
            self.n_observations
        )

        return hessian_likelihood
