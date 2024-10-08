# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

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
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
    ) -> float:
        """Evaluate the likelihood.

        Parameters
        ----------
        eta : ArrayLike
            Vector of the linear predictor.
        y : ArrayLike
            Vector of the observations.

        Returns
        -------
        likelihood : float
            Likelihood.
        """
        pass

    @abstractmethod
    def evaluate_gradient_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        """Evaluate the gradient of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        y : ArrayLike
            Vector of the observations.
        eta : ArrayLike
            Vector of the linear predictor.


        Returns
        -------
        gradient_likelihood : ArrayLike
            Gradient of the likelihood.
        """
        pass

    @abstractmethod
    def evaluate_hessian_likelihood(
        self,
        eta: ArrayLike,
        y: ArrayLike,
        theta_likelihood: dict = None,
    ) -> ArrayLike:
        """Evaluate the Hessian of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : ArrayLike
            Vector of the linear predictor.
        y : ArrayLike
            Vector of the observations.


        Returns
        -------
        hessian_likelihood : ArrayLike
            Hessian of the likelihood.
        """
        pass

    @abstractmethod
    def get_theta(self) -> dict:
        """Get the likelihood initial hyperparameters."""
        pass
