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

        # check that dimensions match


    def construct_Q_prior(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        ## TODO: link this to prior hyperparameters class
        ## i.e. rescale phi and tau according to prior hyperparameters class
        phi = kwargs.get("phi")
        #phi = scaled_logit(phi_scaled, direction="backward")
        #print("phi: ", phi)

        tau = kwargs.get("tau")
        #exp_tau = xp.exp(tau)
        #print("tau: ", tau)

        s2 = 1 / tau
        denom = s2 * (1 - phi**2)

        diag = [(1 + phi**2) / denom] * self.n_latent_parameters
        diag[0] = diag[-1] = 1 / denom
        off_diag = [-phi / denom] * (self.n_latent_parameters - 1)

        Q_prior = sp.sparse.diags([off_diag, diag, off_diag], [-1, 0, 1])
        
        # need this -> otherwise there might be a sorting issue
        Q_prior = Q_prior.tocsr()
        Q_prior.sort_indices()

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
