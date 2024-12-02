# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from scipy.sparse import diags

from pyinla import ArrayLike, xp
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
            e = np.load(pyinla_config.input_dir / "e.npy")
        except FileNotFoundError:
            e = np.ones((n_observations), dtype=int)

        if xp == np:
            self.e = e
        else:
            self.e = xp.asarray(e)

    def evaluate_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> float:
        likelihood = xp.dot(eta, y) - xp.sum(self.e * xp.exp(eta))

        return likelihood

    def evaluate_gradient_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> ArrayLike:
        gradient_likelihood = y - self.e * xp.exp(eta)

        return gradient_likelihood

    def evaluate_hessian_likelihood(
        self,
        **kwargs,
    ) -> ArrayLike:
        eta = kwargs.get("eta", None)

        hessian_likelihood = -diags(self.e * xp.exp(eta))

        return hessian_likelihood
