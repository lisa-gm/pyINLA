# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from pathlib import Path

from pyinla import xp, sp, ArrayLike, NDArray
from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig


class PoissonLikelihood(Likelihood):
    """Poisson likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_observations: int,
    ) -> None:
        """Initializes the Poisson likelihood."""
        super().__init__(pyinla_config, n_observations)

        # Load the extra coeficients for Poisson likelihood
        try:
            e: NDArray = np.load(Path.joinpath(pyinla_config.input_dir, "e.npy"))
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
    ) -> float:
        likelihood: float = xp.dot(eta, y) - xp.sum(self.e * xp.exp(eta))

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
        eta: NDArray = kwargs.get("eta", None)

        hessian_likelihood: ArrayLike = -1.0 * sp.sparse.diags(self.e * xp.exp(eta))

        return hessian_likelihood
