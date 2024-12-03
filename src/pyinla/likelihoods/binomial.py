# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from pathlib import Path

from pyinla import xp, sp, ArrayLike, NDArray
from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.utils import sigmoid


class BinomialLikelihood(Likelihood):
    """Binomial likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_observations: int,
    ) -> None:
        """Initializes the Binomial likelihood."""
        super().__init__(pyinla_config, n_observations)

        # Load the extra coeficients for Binomial likelihood
        try:
            n_trials: NDArray = np.load(
                Path.joinpath(pyinla_config.input_dir, "n_trials.npy")
            )
            if xp == np:
                self.n_trials: NDArray = n_trials
            else:
                self.n_trials: NDArray = xp.asarray(n_trials)
        except FileNotFoundError:
            self.n_trials: NDArray = xp.ones((n_observations), dtype=int)

        if pyinla_config.model.likelihood.link_function == "sigmoid":
            self.link_function = sigmoid
        else:
            raise NotImplementedError(
                f"Link function {pyinla_config.model.likelihood.link_function} not implemented."
            )

    def evaluate_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> float:
        """Evalutate the a binomial likelihood.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.

        Notes
        -----
        For now only a sigmoid link-function is implemented.

        Returns
        -------
        likelihood : float
            Likelihood.
        """
        linkEta: NDArray = self.link_function(eta)

        likelihood: float = xp.dot(y, xp.log(linkEta)) + xp.dot(
            self.n_trials - y, xp.log(1 - linkEta)
        )

        return likelihood

    def evaluate_gradient_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        """
        Evaluate the gradient of the binomial likelihood with respect to eta.

        Parameters
        ----------
        eta : NDArray
            Linear predictor.
        y : NDArray
            Observed data.

        Returns
        -------
        grad_likelihood : NDArray
            Gradient of the likelihood with respect to eta.
        """

        linkEta: NDArray = self.link_function(eta)
        grad_likelihood: NDArray = y - self.n_trials * linkEta

        return grad_likelihood

    def evaluate_hessian_likelihood(
        self,
        **kwargs,
    ) -> ArrayLike:
        """
        Evaluate the Hessian of the binomial likelihood with respect to eta.

        Parameters
        ----------
        eta : NDArray
            Linear predictor.
        y : NDArray
            Observed data.


        Returns
        -------
        hess_likelihood : NDArray
            Hessian of the likelihood with respect to eta.
        """
        eta: NDArray = kwargs.get("eta", None)

        linkEta: NDArray = self.link_function(eta)
        hess_likelihood: ArrayLike = -self.n_trials * linkEta * (1 - linkEta)

        return sp.sparse.diags(hess_likelihood)
