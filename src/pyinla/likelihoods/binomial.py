# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np

from pyinla import ArrayLike, xp, sp
from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.utils.link_functions import sigmoid


class BinomialLikelihood(Likelihood):
    """Binomial likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_observations: int,
    ) -> None:
        """Initializes the Binomial likelihood."""
        super().__init__(pyinla_config, n_observations)

        # load the extra coeficients for Binomial likelihood
        try:
            n_trials = np.load(pyinla_config.input_dir / "n_trials.npy")
            if xp == np:
                self.n_trials = n_trials
            else:
                self.n_trials = xp.asarray(n_trials)
        except FileNotFoundError:
            self.n_trials = xp.ones((n_observations), dtype=int)

        if pyinla_config.likelihood.link_function == "sigmoid":
            self.link_function = sigmoid
        else:
            raise NotImplementedError(
                f"Link function {pyinla_config.likelihood.link_function} not implemented."
            )

    def evaluate_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> float:
        """Evalutate the a binomial likelihood.

        Parameters
        ----------
        eta : ArrayLike
            Vector of the linear predictor.
        y : ArrayLike
            Vector of the observations.

        Notes
        -----
        For now only a sigmoid link-function is implemented.

        Returns
        -------
        likelihood : float
            Likelihood.
        """
        linkEta = self.link_function(eta)

        likelihood = xp.dot(y, xp.log(linkEta)) + xp.dot(
            self.n_trials - y, xp.log(1 - linkEta)
        )

        return likelihood

    def evaluate_gradient_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ) -> ArrayLike:
        """
        Evaluate the gradient of the binomial likelihood with respect to eta.

        Parameters
        ----------
        eta : ArrayLike
            Linear predictor.
        y : ArrayLike
            Observed data.

        Returns
        -------
        grad_likelihood : ArrayLike
            Gradient of the likelihood with respect to eta.
        """

        linkEta = self.link_function(eta)
        grad_likelihood = y - self.n_trials * linkEta

        return grad_likelihood

    def evaluate_hessian_likelihood(
        self,
        **kwargs,
    ) -> ArrayLike:
        """
        Evaluate the Hessian of the binomial likelihood with respect to eta.

        Parameters
        ----------
        eta : ArrayLike
            Linear predictor.
        y : ArrayLike
            Observed data.


        Returns
        -------
        hess_likelihood : ArrayLike
            Hessian of the likelihood with respect to eta.
        """
        eta = kwargs.get("eta", None)

        linkEta = self.link_function(eta)
        hess_likelihood = -self.n_trials * linkEta * (1 - linkEta)

        return sp.sparse.diags(hess_likelihood)
