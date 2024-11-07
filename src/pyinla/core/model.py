# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

# from pyinla import ArrayLike
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
    def get_theta(self) -> dict:
        """Get the initial theta of the model. This dictionary is constructed
        at instanciation of the model. It has to be stored in the model as
        theta is specific to the model.

        Returns
        -------
        theta_inital_model : dict
            Dictionary of initial hyperparameters.
        """
        ...

    @abstractmethod
    def construct_Q_prior(self, theta_model: dict = None) -> sparray:
        """Construct the prior precision matrix."""
        pass

    @abstractmethod
    def construct_Q_conditional(
        self,
        Q_prior: sparray,
        a: sparray,
        hessian_likelihood: sparray,
    ) -> sparray:
        """Construct the conditional precision matrix.
        #TODO: hessian_likelihood always diagonal (for all models). How to pass this best?
        """
        pass
