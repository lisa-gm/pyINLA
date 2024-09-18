# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike
from scipy.sparse import sparray

from pyinla.core.pyinla_config import PyinlaConfig


class Likelihood(ABC):
    """Abstract core class for likelihood."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_observations: int,
    ) -> None:
        """Initializes the likelihood."""

        self.pyinla_config = pyinla_config
        self.n_observations = n_observations

    @abstractmethod
    def evaluate_likelihood(
        self,
        y: ArrayLike,
        a: sparray,
        x: ArrayLike,
        theta_likelihood: dict = None,
    ) -> float:
        """Evaluate the likelihood.

        Parameters
        ----------
        y : ArrayLike
            Vector of the observations.
        a : sparray
            Design matrix.
        x : ArrayLike
            Vector of the latent parameters.

        Returns
        -------
        likelihood : float
            Likelihood.
        """
        pass

    @abstractmethod
    def get_theta_initial(self) -> dict:
        """Get the likelihood initial hyperparameters."""
        pass
