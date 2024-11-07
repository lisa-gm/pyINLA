# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from autograd.numpy import dot, log
from numpy.typing import ArrayLike
from scipy.sparse import diags

from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.utils.link_functions import sigmoid

# from scipy.sparse import sparray


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
            self.n_trials = np.load(pyinla_config.input_dir / "n_trials.npy")
        except FileNotFoundError:
            self.n_trials = np.ones((n_observations), dtype=int)

        if pyinla_config.likelihood.link_function == "sigmoid":
            self.link_function = sigmoid
        else:
            raise NotImplementedError(
                f"Link function {pyinla_config.likelihood.link_function} not implemented."
            )

    def get_theta(self) -> dict:
        """Get the likelihood initial hyperparameters."""
        return {}

    def evaluate_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
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

        likelihood = np.dot(y, np.log(linkEta)) + np.dot(
            self.n_trials - y, np.log(1 - linkEta)
        )

        return likelihood

    def evaluate_likelihood_autodiff(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
    ) -> float:
        """Evalutate the a binomial likelihood. VERSION FOR AUTOGRAD PACKAGE.

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

        likelihood = dot(y, log(linkEta)) + dot(self.n_trials - y, log(1 - linkEta))

        return likelihood

    def evaluate_gradient_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        """
        Evaluate the gradient of the binomial likelihood with respect to eta.

        Parameters
        ----------
        eta : ArrayLike
            Linear predictor.
        y : ArrayLike
            Observed data.
        theta_likelihood : dict, optional
            Dictionary of likelihood hyperparameters.

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
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        """
        Evaluate the Hessian of the binomial likelihood with respect to eta.

        Parameters
        ----------
        eta : ArrayLike
            Linear predictor.
        y : ArrayLike
            Observed data.
        theta_likelihood : dict, optional
            Dictionary of likelihood hyperparameters.

        Returns
        -------
        hess_likelihood : ArrayLike
            Hessian of the likelihood with respect to eta.
        """

        linkEta = self.link_function(eta)
        hess_likelihood = -self.n_trials * linkEta * (1 - linkEta)

        return diags(hess_likelihood)
