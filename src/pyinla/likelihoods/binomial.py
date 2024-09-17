# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import sp_array

from pyinla.core.likelihood import Likelihood
from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.utils.link_functions import sigmoid


class BinomialLikelihood(Likelihood):
    """Binomial likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_observations: int,
        **kwargs,
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

    def evaluate_likelihood(
        self,
        y: ArrayLike,
        a: sp_array,
        x: ArrayLike,
        **kwargs,
    ) -> float:
        sig = self.link_function(a @ x)
        likelihood = np.dot(y, np.log(sig)) + np.dot(self.n_trials - y, np.log(1 - sig))

        return likelihood
