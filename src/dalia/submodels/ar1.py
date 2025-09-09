# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate

from dalia import sp, xp
from dalia.configs.submodels_config import AR1SubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class AR1SubModel(SubModel):
    """Fit an AR(1) model."""

    def __init__(
        self,
        config: AR1SubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        self.n_latent_parameters: int = config.n_latent_parameters

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        s2 = kwargs.get("s2")
        phi = kwargs.get("phi")
        denom = s2 * (1 - phi**2)

        diag = [(1 + phi**2) / denom] * self.n_latent_parameters
        diag[0] = diag[-1] = 1 / denom
        off_diag = [-phi / denom] * (self.n_latent_parameters - 1)

        self.Q_prior = sp.diags([diag, off_diag, off_diag], [0, -1, 1])

        return self.Q_prior

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Submodel Type", self.submodel_type],
            ["Number of Latent Parameters", self.n_latent_parameters],
        ]
        submodel_table = tabulate(
            values,
            tablefmt="fancy_grid",
            colalign=("left", "center"),
        )

        # Add the header title
        submodel_table = add_str_header(
            title=self.submodel_type.replace("_", " ").title(),
            table=submodel_table,
        )
        str_representation += submodel_table

        return str_representation


if __name__ == "__main__":

    n = 5

    s2 = 1
    phi = xp.sqrt(0.5)
