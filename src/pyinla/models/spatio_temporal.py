# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from scipy.sparse import csc_matrix, kron, load_npz, sparray

from pyinla.core.model import Model
from pyinla.core.pyinla_config import PyinlaConfig


class SpatioTemporalModel(Model):
    """Fit a spatio-temporal model."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        n_latent_parameters: int,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        super().__init__(pyinla_config, n_latent_parameters)

        # Load spatial_matrices
        self.c0 = load_npz(pyinla_config.input_dir / "c0.npz")
        self.g1 = load_npz(pyinla_config.input_dir / "g1.npz")
        self.g2 = load_npz(pyinla_config.input_dir / "g2.npz")
        self.g3 = load_npz(pyinla_config.input_dir / "g3.npz")

        self.ns = self.c0.shape[0]
        self._check_dimensions_spatial_matrices()

        # Load temporal_matrices
        self.m0 = load_npz(pyinla_config.input_dir / "m0.npz")
        self.m1 = load_npz(pyinla_config.input_dir / "m1.npz")
        self.m2 = load_npz(pyinla_config.input_dir / "m2.npz")

        self.nt = self.m0.shape[0]
        self._check_dimensions_temporal_matrices()

        # Check that design_matrix shape match spatio-temporal fields
        assert (
            self.n_latent_parameters == self.nb + self.ns * self.nt
        ), f"Design matrix has incorrect number of columns. \n    n_latent_parameters: {self.n_latent_parameters}\n    nb: {self.nb} + ns: {self.ns} * nt: {self.nt} = {self.nb + self.ns * self.nt}"

        # Load model hyperparameters
        self.theta = {
            "spatial_range": pyinla_config.model.theta_spatial_range,
            "temporal_range": pyinla_config.model.theta_temporal_range,
            "spatio_temporal_variation": pyinla_config.model.theta_spatio_temporal_variation,
        }

    def _check_dimensions_spatial_matrices(self) -> None:
        """Check the dimensions of the model."""
        # Spatial matrices checks
        assert self.c0.shape[0] == self.c0.shape[1], "Spatial matrix c0 is not square."
        assert self.g1.shape[0] == self.g1.shape[1], "Spatial matrix g1 is not square."
        assert self.g2.shape[0] == self.g2.shape[1], "Spatial matrix g2 is not square."
        assert self.g3.shape[0] == self.g3.shape[1], "Spatial matrix g3 is not square."
        assert (
            self.c0.shape == self.g1.shape == self.g2.shape == self.g3.shape
        ), "Dimensions of spatial matrices do not match."

    def _check_dimensions_temporal_matrices(self) -> None:
        """Check the dimensions of the model."""
        # Temporal matrices checks
        assert self.m0.shape[0] == self.m0.shape[1], "Temporal matrix m0 is not square."
        assert self.m1.shape[0] == self.m1.shape[1], "Temporal matrix m1 is not square."
        assert self.m2.shape[0] == self.m2.shape[1], "Temporal matrix m2 is not square."
        assert (
            self.m0.shape == self.m1.shape == self.m2.shape
        ), "Dimensions of temporal matrices do not match."

    def get_theta(self) -> dict:
        """Get the initial theta of the model. This dictionary is constructed
        at instanciation of the model. It has to be stored in the model as
        theta is specific to the model.

        Returns
        -------
        theta_inital_model : dict
            Dictionary of hyperparameters. Theta gets when calling construct_Q_prior.
        """
        return self.theta

    def construct_Q_prior(self, theta_model: dict = None) -> sparray:
        """Construct the prior precision matrix."""

        self.theta = theta_model

        if theta_model is None:
            raise ValueError("theta_model must be provided.")

        theta_spatial_range = np.exp(theta_model["spatial_range"])
        theta_temporal_range = np.exp(theta_model["temporal_range"])
        theta_spatio_temporal_variation = np.exp(
            theta_model["spatio_temporal_variation"]
        )
        # print("theta_spatial_range: ", theta_spatial_range)
        # print("theta_temporal_range: ", theta_temporal_range)
        # print("theta_spatio_temporal_variation: ", theta_spatio_temporal_variation)

        q1s = pow(theta_spatial_range, 2) * self.c0 + self.g1
        q2s = (
            pow(theta_spatial_range, 4) * self.c0
            + 2 * pow(theta_spatial_range, 2) * self.g1
            + self.g2
        )
        q3s = (
            pow(theta_spatial_range, 6) * self.c0
            + 3 * pow(theta_spatial_range, 4) * self.g1
            + 3 * pow(theta_spatial_range, 2) * self.g2
            + self.g3
        )

        Q_spatio_temporal = pow(theta_spatio_temporal_variation, 2) * (
            kron(self.m0, q3s)
            + theta_temporal_range * kron(self.m1, q2s)
            + pow(theta_temporal_range, 2) * kron(self.m2, q1s)
        )

        if Q_spatio_temporal is not csc_matrix:
            Q_spatio_temporal = csc_matrix(Q_spatio_temporal)

        # Construct block diagonal matrix Q fixed effects
        Q_fixed_effects_data = np.full(self.nb, self.fixed_effects_prior_precision)
        Q_fixed_effects_indices = np.arange(self.nb)
        Q_fixed_effects_indptr = np.arange(self.nb + 1)

        # Extract data, indices, and indptr from Q_spatio_temporal
        Q_spatio_temporal_data = Q_spatio_temporal.data
        Q_spatio_temporal_indices = Q_spatio_temporal.indices
        Q_spatio_temporal_indptr = Q_spatio_temporal.indptr
        Q_spatio_temporal_shape = Q_spatio_temporal.shape

        # Concatenate data, indices, and indptr to form the block diagonal matrix Q
        data = np.concatenate([Q_spatio_temporal_data, Q_fixed_effects_data])
        indices = np.concatenate(
            [
                Q_spatio_temporal_indices,
                Q_fixed_effects_indices + Q_spatio_temporal_shape[1],
            ]
        )
        indptr = np.concatenate(
            [
                Q_spatio_temporal_indptr,
                Q_spatio_temporal_indptr[-1] + Q_fixed_effects_indptr[1:],
            ]
        )

        Q_prior = csc_matrix(
            (data, indices, indptr),
            shape=(
                Q_spatio_temporal.shape[0] + self.nb,
                Q_spatio_temporal.shape[1] + self.nb,
            ),
        )

        return Q_prior

    def construct_Q_conditional(
        self,
        Q_prior: sparray,
        a: sparray,
        hessian_likelihood: sparray,
    ) -> float:
        """Construct the conditional precision matrix."""

        Q_conditional = Q_prior - a.T @ hessian_likelihood @ a

        return Q_conditional
