# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import sp
from pyinla.configs.submodels_config import RegressionSubModelConfig
from pyinla.core.submodel import SubModel


class RegressionSubModel(SubModel):
    """Fit a regression model."""

    def __init__(
        self,
        config: RegressionSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        self.n_fixed_effects: int = config.n_fixed_effects
        self.fixed_effects_prior_precision: float = config.fixed_effects_prior_precision

        # Check that design_matrix shape match number of fixed effects
        assert (
            self.n_fixed_effects == self.n_latent_parameters
        ), f"Design matrix has {self.n_latent_parameters} columns, but expected {self.n_fixed_effects} columns."

        # --- Construct the prior precision matrix
        self.Q_prior: sp.sparse.coo_matrix = sp.sparse.coo_matrix(
            self.fixed_effects_prior_precision * sp.sparse.eye(self.n_fixed_effects)
        )

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        return self.Q_prior

    def __str__(self) -> str:
        """String representation of the submodel."""
        return (
            " --- RegressionSubModel ---\n"
            f"  n_fixed_effects: {self.n_fixed_effects}\n"
            f"  fixed_effects_prior_precision: {self.fixed_effects_prior_precision}"
        )
