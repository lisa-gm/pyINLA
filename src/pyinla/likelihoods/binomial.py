# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import sparray

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
            self.n_trials = np.load(pyinla_config.input_dir / "n_trials.npy")
        except FileNotFoundError:
            self.n_trials = np.ones((n_observations), dtype=int)

        if pyinla_config.likelihood.link_function == "sigmoid":
            self.link_function = sigmoid

    def get_theta_initial(self) -> dict:
        """Get the likelihood initial hyperparameters."""
        return {}

    def evaluate_likelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        theta_likelihood: dict = None,
    ) -> float:

        # hardcoded link function for now: ONLY sigmoid
        linkEta = sigmoid(eta)
        # dot(y, log(linkEta)) + dot(Ntrials - y, log(1 - linkEta))
        likelihood = np.dot(
            y, np.log(linkEta) + np.dot(self.n_trials - y, np.log(1 - linkEta))
        )

        return likelihood

    def evaluate_gradient_likelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        raise NotImplementedError

    def evaluate_hessian_likelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        raise NotImplementedError
