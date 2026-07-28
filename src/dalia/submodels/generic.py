# Copyright 2024-2026 DALIA authors. All rights reserved.
from tabulate import tabulate

import numpy as np
from scipy.sparse import load_npz, spmatrix

from dalia import sp, xp, NDArray
from dalia.configs.submodels_config import GenericSubModelConfig
from dalia.core.submodel import SubModel
from dalia.utils import add_str_header


class GenericSubModel(SubModel):
    """Fit a generic model meaning the precision matrix of the latent parameters is directly specified by the user. The mean is assumed to be zero."""

    def __init__(
        self,
        config: GenericSubModelConfig,
    ) -> None:
        """Initializes the model."""
        super().__init__(config)

        self.tau = self.config.tau

        # accept sparse or dense input for the precision matrix
        try:
            q: spmatrix = load_npz(self.input_path.joinpath("q.npz"))
            self.q = sp.sparse.csc_matrix(q)
        except FileNotFoundError:
            # check if dense q matrix exists
            try:
                q: NDArray = np.load(self.input_path.joinpath("q.npy"))
                if xp == np:
                    self.q: NDArray = q
                else:
                    self.q: NDArray = xp.array(q)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "No precision matrix found for generic model. Please provide a valid precision matrix."
                )

        # Check that q shape match number of latent parameters
        assert (
            self.q.shape[0] == self.q.shape[1] == self.n_latent_parameters
        ), f"Precision matrix has {self.q.shape[0]} rows and {self.q.shape[1]} columns, but expected {self.n_latent_parameters} rows and columns."

        print(
            f"Successfully loaded precision matrix of shape {self.q.shape} for generic model."
        )

    def _construct_Q_prior_core(self, **kwargs) -> sp.sparse.coo_matrix:
        """Construct the prior precision matrix."""

        tau = kwargs.get("tau")
        self.Q_prior = tau * self.q

        return self.Q_prior.tocoo()

    def __str__(self) -> str:
        """String representation of the submodel."""
        str_representation = ""

        # --- Make the Submodel table ---
        values = [
            ["Submodel Type", self.submodel_type],
            ["Number of Latent Effects", self.n_latent_parameters],
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
