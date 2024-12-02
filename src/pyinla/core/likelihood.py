# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from pyinla import ArrayLike
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
        **kwargs,
    ) -> float:
        """Evaluate the likelihood.

        Parameters
        ----------
        eta : ArrayLike
            Vector of the linear predictor.
        y : ArrayLike
            Vector of the observations.
        **kwargs : optional
            Hyperparameters for likelihood.


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
        **kwargs,
    ) -> ArrayLike:
        """Evaluate the gradient of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        y : ArrayLike
            Vector of the observations.
        eta : ArrayLike
            Vector of the linear predictor.
        **kwargs : optional
            Hyperparameters for likelihood.

        Returns
        -------
        gradient_likelihood : ArrayLike
            Gradient of the likelihood.
        """
        pass

    @abstractmethod
    def evaluate_hessian_likelihood(
        self,
        **kwargs,
    ) -> ArrayLike:
        """Evaluate the Hessian of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : ArrayLike
            Vector of the linear predictor.
        y : ArrayLike
            Vector of the observations.
        **kwargs : optional
            Hyperparameters for likelihood.


        Returns
        -------
        hessian_likelihood : ArrayLike
            Hessian of the likelihood.
        """
        pass
