# Copyright 2024-2025 DALIA authors. All rights reserved.
from tabulate import tabulate

import numpy as np

from dalia import sp, xp
from dalia.configs.submodels_config import AR1SubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header, scaled_logit


class AR1SubModel(SubModel):
    """Fit an AR(1) model."""

    def __init__(
        self,
        config: AR1SubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        tau = kwargs.get("tau")
        exp_tau = xp.exp(tau)
        phi_scaled = kwargs.get("phi")
        # print("tau:", exp_tau)
        #print("phi_scaled:", phi_scaled)
        phi = scaled_logit(phi_scaled, direction="backward")
        print("phi:", phi)
        s2 = 1 / exp_tau
        denom = s2 * (1 - phi**2)

        diag = [(1 + phi**2) / denom] * self.n_latent_parameters
        diag[0] = diag[-1] = 1 / denom
        off_diag = [-phi / denom] * (self.n_latent_parameters - 1)

        ## TODO: how to do this more efficiently?
        Q_prior = sp.sparse.diags([off_diag, diag, off_diag], [-1, 0, 1])
        Q_prior = Q_prior.tocsr()
        Q_prior.sort_indices()
        # print("Q_prior.data: ", Q_prior.data)

        # print("Qprior: \n", Q_prior.toarray())

        return Q_prior.tocoo()

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Submodel Type", self.submodel_type],
            ["Number of Latent Parameters", self.n_latent_parameters],
            ["Phi", f"{self.config.phi:.3f}"],
            ["tau", f"{self.config.tau:.3f}"],
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
