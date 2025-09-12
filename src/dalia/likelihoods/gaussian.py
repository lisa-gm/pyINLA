# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import ArrayLike, NDArray, sp, xp
from dalia.configs.likelihood_config import GaussianLikelihoodConfig
from dalia.core.likelihood import Likelihood

try:
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    JAX_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    jnp = xp
    JAX_AVAILABLE = False


class GaussianLikelihood(Likelihood):
    """Gaussian likelihood."""

    def __init__(
        self,
        n_observations: int,
        config: GaussianLikelihoodConfig,
    ) -> None:
        """Initializes the Gaussian likelihood."""
        super().__init__(n_observations, config)

        if JAX_AVAILABLE:
            first_derivative = grad(self.evaluate_likelihood_jax, argnums=0)
            second_derivative = grad(first_derivative, argnums=0)
            self.gradient_jax = jit(vmap(first_derivative))
            self.hessian_jax = jit(vmap(second_derivative))

    def evaluate_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
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

        likelihood = 0.5 * theta - 0.5 * xp.exp(theta) * yEta * yEta
        

        return likelihood
    
    def evaluate_likelihood_jax(self, eta, y, theta):
        yEta = eta - y
        return 0.5 * theta - 0.5 * jnp.exp(theta) * yEta * yEta

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
    
    def evaluate_gradient_likelihood_jax(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        jax_eta = jnp.from_dlpack(eta)
        jax_y = jnp.from_dlpack(y)
        theta = kwargs.get("theta", None)
        if not isinstance(theta, float):
            theta = float(theta[0])
        jax_theta = jnp.full_like(jax_eta, theta)
        grad = self.gradient_jax(jax_eta, jax_y, jax_theta)
        return xp.from_dlpack(grad)

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
    
    def evaluate_hessian_likelihood_jax(
        self,
        **kwargs,
    ) -> ArrayLike:
        jax_eta = jnp.from_dlpack(kwargs.get("eta"))
        jax_y = jnp.from_dlpack(kwargs.get("y"))
        theta = kwargs.get("theta", None)
        if not isinstance(theta, float):
            theta = float(theta[0])
        jax_theta = jnp.full_like(jax_eta, theta)
        hessian = self.hessian_jax(jax_eta, jax_y, jax_theta)
        return xp.from_dlpack(hessian)
