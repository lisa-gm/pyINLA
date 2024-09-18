# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike
from scipy.sparse import sparray

from pyinla.core.pyinla_config import PyinlaConfig


class Model(ABC):
    """Abstract core class for statistical models."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_latent_parameters: int,
    ) -> None:
        """Initializes the model."""
        self.pyinla_config = pyinla_config
        self.nb = self.pyinla_config.model.n_fixed_effects
        self.fixed_effects_prior_precision = (
            pyinla_config.model.fixed_effects_prior_precision
        )

        self.n_latent_parameters = n_latent_parameters

    @abstractmethod
    def get_theta_initial(self) -> dict:
        """Get the model initial hyperparameters."""
        pass

    @abstractmethod
    def construct_Q_prior(self, theta_model: dict = None) -> float:
        """Construct the prior precision matrix."""
        pass

    @abstractmethod
    def construct_Q_conditional(
        self,
        Q_prior: sparray,
        y: ArrayLike,
        a: sparray,
        x: ArrayLike,
        theta_model: dict = None,
    ) -> float:
        """Construct the conditional precision matrix."""
        pass
