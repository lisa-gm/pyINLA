# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from pyinla import ArrayLike, NDArray
from pyinla.configs.likelihood_config import LikelihoodConfig


class Likelihood(ABC):
    """Abstract core class for likelihood."""

    def __init__(
        self,
        n_observations: int,
        config: LikelihoodConfig,
    ) -> None:
        """Initializes the likelihood."""

        self.config = config
        self.n_observations = n_observations

    @abstractmethod
    def evaluate_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> float:
        """Evaluate the likelihood.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
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
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        """Evaluate the gradient of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        y : NDArray
            Vector of the observations.
        eta : NDArray
            Vector of the linear predictor.
        **kwargs : optional
            Hyperparameters for likelihood.

        Returns
        -------
        gradient_likelihood : NDArray
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
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        **kwargs : optional
            Hyperparameters for likelihood.


        Returns
        -------
        hessian_likelihood : ArrayLike
            Hessian of the likelihood.
        """
        pass
