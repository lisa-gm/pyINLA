# Copyright 2024 pyINLA authors. All rights reserved.

# from pyinla import ArrayLike
from pyinla import sparse
from scipy.sparse import sparray

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
        self.fixed_effects_prior_precision = (
            submodel_config.fixed_effects_prior_precision
        )

        # Check that design_matrix shape match number of fixed effects
        assert (
            self.n_fixed_effects == self.n_latent_parameters
        ), "Design matrix has incorrect number of columns."

        # --- Construct the prior precision matrix
        self.Q_prior = self.fixed_effects_prior_precision * sparse.eye(
            self.n_fixed_effects
        )

    def construct_Q_prior(self, **kwargs) -> sparray:
        """Construct the prior precision matrix."""

        return self.Q_prior

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
