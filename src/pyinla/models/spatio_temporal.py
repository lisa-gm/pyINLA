# Copyright 2024 pyINLA authors. All rights reserved.

import math

import numpy as np
from cupyx.profiler import time_range
from scipy.sparse import csc_matrix, kron, load_npz, sparray
from scipy.special import gamma

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

    @time_range()
    def convert_theta_from_interpret2model(
        self, theta_interpret: dict, dim_spatial_domain=2, manifold="plane"
    ) -> dict:
        """Convert theta from interpretable scale to model scale."""

        if dim_spatial_domain != 2:
            raise ValueError("Only 2D spatial domain is supported for now.")

        # assumes alphas as fixed for now
        alpha_s = 2
        alpha_t = 1
        alpha_e = 1

        # implicit assumption that spatial domain is 2D
        alpha = alpha_e + alpha_s * (alpha_t - 0.5)

        nu_s = alpha - 1
        nu_t = alpha_t - 0.5

        log_gamma_s = 0.5 * np.log(8 * nu_s) - theta_interpret["spatial_range"]
        log_gamma_t = (
            theta_interpret["temporal_range"]
            - 0.5 * np.log(8 * nu_t)
            + alpha_s * log_gamma_s
        )

        if manifold == "sphere":
            cR_t = gamma(nu_t) / (gamma(alpha_t) * pow(4 * math.pi, 0.5))
            cS = 0.0
            for k in range(50):
                cS += (2.0 * k + 1.0) / (
                    4.0
                    * math.pi
                    * pow(pow(np.exp(log_gamma_s), 2) + k * (k + 1), alpha)
                )
            log_gamma_e = (
                0.5 * np.log(cR_t)
                + 0.5 * np.log(cS)
                - 0.5 * log_gamma_t
                - theta_interpret["spatio_temporal_variation"]
            )

        elif manifold == "plane":
            #  pow(4*M_PI, dim_spatial_domain/2.0) * pow(4*M_PI, dim_temporal_domain/2.0);
            c1_scaling_constant = pow(4 * math.pi, 1.5)
            c1 = (
                gamma(nu_t)
                * gamma(nu_s)
                / (gamma(alpha_t) * gamma(alpha) * c1_scaling_constant)
            )
            log_gamma_e = (
                0.5 * np.log(c1)
                - 0.5 * log_gamma_t
                - nu_s * log_gamma_s
                - theta_interpret["spatio_temporal_variation"]
            )
        else:
            raise ValueError("Manifold not supported: ", manifold)

        theta_model = {
            "spatial_range": log_gamma_s,
            "temporal_range": log_gamma_t,
            "spatio_temporal_variation": log_gamma_e,
        }

        return theta_model

    @time_range()
    def convert_theta_from_model2interpret(
        self, theta_model: dict, dim_spatial_domain=2, manifold="plane"
    ) -> dict:
        """Convert theta from model scale to interpretable scale."""

        if dim_spatial_domain != 2:
            raise ValueError("Only 2D spatial domain is supported for now.")

        alpha_t = 1
        alpha_s = 2
        alpha_e = 1

        alpha = alpha_e + alpha_s * (alpha_t - 0.5)
        nu_s = alpha - 1
        nu_t = alpha_t - 0.5

        gamma_s = np.exp(theta_model["spatial_range"])
        gamma_t = np.exp(theta_model["temporal_range"])
        gamma_e = np.exp(theta_model["spatio_temporal_variation"])

        theta_interpret = {}
        theta_interpret["spatial_range"] = np.log(np.sqrt(8 * nu_s) / gamma_s)
        theta_interpret["temporal_range"] = np.log(
            gamma_t * np.sqrt(8 * nu_t) / (pow(gamma_s, alpha_s))
        )

        if manifold == "sphere":
            cR_t = gamma(alpha_t - 0.5) / (gamma(alpha_t) * pow(4 * math.pi, 0.5))
            cS = 0.0
            for k in range(50):  # compute 1st 50 terms of infinite sum
                cS += (2.0 * k + 1) / (
                    4 * math.pi * pow(pow(gamma_s, 2) + k * (k + 1), alpha)
                )
            # print(f"cS : {cS}")
            theta_interpret["spatio_temporal_variation"] = np.log(
                np.sqrt(cR_t * cS) / (gamma_e * np.sqrt(gamma_t))
            )

        elif manifold == "plane":
            c1_scaling_const = pow(4 * math.pi, dim_spatial_domain / 2.0) * pow(
                4 * math.pi, 0.5
            )
            c1 = (
                gamma(nu_t)
                * gamma(nu_s)
                / (gamma(alpha_t) * gamma(alpha) * c1_scaling_const)
            )
            theta_interpret["spatio_temporal_variation"] = np.log(
                np.sqrt(c1)
                / (
                    (gamma_e * np.sqrt(gamma_t))
                    * pow(gamma_s, alpha - dim_spatial_domain / 2)
                )
            )

        else:
            raise ValueError("Manifold not supported: ", manifold)

        return theta_interpret

    @time_range()
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

        # with time_range("sparseKroneckerProduct", color_id=0):
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

    @time_range()
    def construct_Q_conditional(
        self,
        Q_prior: sparray,
        a: sparray,
        hessian_likelihood: sparray,
    ) -> float:
        """Construct the conditional precision matrix.

        Note
        ----
        Input of the hessian of the likelihood is a diagonal matrix.
        The negative hessian is required, therefore the minus in front.

        """

        Q_conditional = Q_prior - a.T @ hessian_likelihood @ a

        return Q_conditional
