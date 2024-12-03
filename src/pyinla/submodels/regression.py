# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla import sp

from pyinla.core.submodel import SubModel
from pyinla.core.pyinla_config import RegressionSubModelConfig
from pathlib import Path


class RegressionModel(SubModel):
    """Fit a regression model."""

    def __init__(
        self,
        submodel_config: RegressionSubModelConfig,
        simulation_path: Path,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        super().__init__(submodel_config, simulation_path)

        self.n_fixed_effects: int = submodel_config.n_fixed_effects
        self.fixed_effects_prior_precision: float = (
            submodel_config.fixed_effects_prior_precision
        )

        # Check that design_matrix shape match number of fixed effects
        assert (
            self.n_fixed_effects == self.n_latent_parameters
        ), "Design matrix has incorrect number of columns."

        # --- Construct the prior precision matrix
        self.Q_prior: sp.sparse.spmatrix = (
            self.fixed_effects_prior_precision * sp.sparse.eye(self.n_fixed_effects)
        )

    def construct_Q_prior(self, **kwargs) -> sp.sparse.spmatrix:
        """Construct the prior precision matrix."""

        return self.Q_prior
