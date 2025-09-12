# Copyright 2024-2025 DALIA authors. All rights reserved.

from pathlib import Path

import numpy as np

from dalia import ArrayLike, NDArray, sp, xp
from dalia.configs.likelihood_config import PoissonLikelihoodConfig
from dalia.core.likelihood import Likelihood

try:
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    JAX_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    jnp = xp
    JAX_AVAILABLE = False


class PoissonLikelihood(Likelihood):
    """Poisson likelihood."""

    def __init__(
        self,
        n_observations: int,
        config: PoissonLikelihoodConfig,
    ) -> None:
        """Initializes the Poisson likelihood."""
        super().__init__(n_observations, config)

        # Load the extra coeficients for Poisson likelihood
        try:
            e: NDArray = np.load(Path(config.input_dir).joinpath("e.npy"))

        except FileNotFoundError:
            e: NDArray = np.ones((n_observations), dtype=int)

        if xp == np:
            self.e: NDArray = e
        else:
            self.e: NDArray = xp.asarray(e)
        
        if JAX_AVAILABLE:
            first_derivative = grad(self.evaluate_likelihood_jax, argnums=0)
            second_derivative = grad(first_derivative, argnums=0)
            self.gradient_jax = jit(vmap(first_derivative))
            self.hessian_jax = jit(vmap(second_derivative))
            self.jax_e = jnp.from_dlpack(self.e)

    def evaluate_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        # likelihood: float = xp.dot(eta, y) - xp.sum(self.e * xp.exp(eta))
        likelihood = eta * y - self.e * xp.exp(eta)

        return likelihood
    
    def evaluate_likelihood_jax(self, eta, y, e):
        return eta * y - e * jnp.exp(eta)

    def evaluate_gradient_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        gradient_likelihood: NDArray = y - self.e * xp.exp(eta)

        return gradient_likelihood
    
    def evaluate_gradient_likelihood_jax(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        jax_eta = jnp.from_dlpack(eta)
        jax_y = jnp.from_dlpack(y)
        grad = self.gradient_jax(jax_eta, jax_y, self.jax_e)
        return xp.from_dlpack(grad)

    def evaluate_hessian_likelihood(
        self,
        **kwargs,
    ) -> ArrayLike:
        eta: NDArray = kwargs.get("eta")

        hessian_likelihood: ArrayLike = -1.0 * sp.sparse.diags(self.e * xp.exp(eta))

        return hessian_likelihood
    
    def evaluate_hessian_likelihood_jax(
        self,
        **kwargs,
    ) -> ArrayLike:
        jax_eta = jnp.from_dlpack(kwargs.get("eta"))
        jax_y = jnp.from_dlpack(kwargs.get("y"))
        hessian = self.hessian_jax(jax_eta, jax_y, self.jax_e)
        return xp.from_dlpack(hessian)
