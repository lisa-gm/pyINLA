# Copyright 2024 pyINLA authors. All rights reserved.

# from pyinla import ArrayLike
from scipy.sparse import eye, sparray

from pyinla.core.submodel import SubModel
from pyinla.core.pyinla_config import SubModelConfig
from pathlib import Path


class RegressionModel(SubModel):
    """Fit a regression model."""

    def __init__(
        self,
        submodel_config: SubModelConfig,
        simulation_path: Path,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        super().__init__(submodel_config, simulation_path)

        self.n_fixed_effects = submodel_config.n_fixed_effects

        # Check that design_matrix shape match number of fixed effects
        assert (
            self.n_fixed_effects == self.n_latent_parameters
        ), "Design matrix has incorrect number of columns."

    def construct_Q_prior(self, theta_model: dict = None) -> sparray:
        """Construct the prior precision matrix."""

        # Construct the prior precision matrix
        Q_prior = self.fixed_effects_prior_precision * eye(self.n_fixed_effects)

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
