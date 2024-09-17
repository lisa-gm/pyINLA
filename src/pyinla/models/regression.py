# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla.core.model import Model
from pyinla.core.pyinla_config import PyinlaConfig


class Regression(Model):
    """Fit a regression model."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_latent_parameters: int,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        super().__init__(pyinla_config, n_latent_parameters)

        # Check that design_matrix shape match number of fixed effects
        assert (
            self.nb == self.n_latent_parameters
        ), "Design matrix has incorrect number of columns."

    def get_theta_initial(self) -> dict:
        """Get the model hyperparameters."""
        return {}
