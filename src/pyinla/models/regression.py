# Copyright 2024 pyINLA authors. All rights reserved.

# from numpy.typing import ArrayLike
from scipy.sparse import eye, sparray

from pyinla.core.model import Model
from pyinla.core.pyinla_config import PyinlaConfig


class RegressionModel(Model):
    """Fit a regression model."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_latent_parameters: int,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        super().__init__(pyinla_config, n_latent_parameters)

        self.ns = 0
        self.nt = 0

        # Check that design_matrix shape match number of fixed effects
        assert (
            self.nb == self.n_latent_parameters
        ), "Design matrix has incorrect number of columns."

    def get_theta(self) -> dict:
        """Get the initial theta of the model. This dictionary is constructed
        at instanciation of the model. It has to be stored in the model as
        theta is specific to the model.

        Returns
        -------
        theta_inital_model : dict
            Dictionary of initial hyperparameters.
        """
        return {}

    def construct_Q_prior(self, theta_model: dict = None) -> sparray:
        """Construct the prior precision matrix."""

        # Construct the prior precision matrix
        Q_prior = self.fixed_effects_prior_precision * eye(self.nb)

        return Q_prior

    def construct_Q_conditional(
        self,
        Q_prior: sparray,
        a: sparray,
        hessian_likelihood: sparray,
    ) -> sparray:
        """Construct the conditional precision matrix."""

        # TODO: think about where the negative comes in ...
        Q_conditional = Q_prior - a.T @ hessian_likelihood @ a

        return Q_conditional
