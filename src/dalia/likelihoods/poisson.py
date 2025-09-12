# Copyright 2024-2025 DALIA authors. All rights reserved.

from pathlib import Path

import numpy as np

from dalia import ArrayLike, NDArray, sp, xp
from dalia.configs.likelihood_config import PoissonLikelihoodConfig
from dalia.core.likelihood import Likelihood


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

    def evaluate_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        # likelihood: float = xp.dot(eta, y) - xp.sum(self.e * xp.exp(eta))
        likelihood = eta * y - self.e * xp.exp(eta)

        return likelihood

    def evaluate_gradient_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        gradient_likelihood: NDArray = y - self.e * xp.exp(eta)

        return gradient_likelihood

    def evaluate_hessian_likelihood(
        self,
        **kwargs,
    ) -> ArrayLike:
        eta: NDArray = kwargs.get("eta")

        hessian_likelihood: ArrayLike = -1.0 * sp.sparse.diags(self.e * xp.exp(eta))

        return hessian_likelihood
