# Copyright 2024 pyINLA authors. All rights reserved.

import math

import numpy as xp
from scipy.sparse import csc_matrix, kron, load_npz, sparray

from pyinla.core.submodel import SubModel
from pyinla.core.pyinla_config import SubModelConfig
from pathlib import Path

from pyinla import ArrayLike, xp, sp


class SpatioTemporalModel(SubModel):
    """Fit a spatio-temporal model."""

    def __init__(
        self,
        submodel_config: SubModelConfig,
        simulation_path: Path,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        super().__init__(submodel_config, simulation_path)

        # Load spatial_matrices
        self.c0: sparray = csc_matrix(
            load_npz(Path.joinpath(simulation_path, submodel_config.inputs, "c0.npz"))
        )
        self.g1: sparray = csc_matrix(
            load_npz(Path.joinpath(simulation_path, submodel_config.inputs, "g1.npz"))
        )
        self.g2: sparray = csc_matrix(
            load_npz(Path.joinpath(simulation_path, submodel_config.inputs, "g2.npz"))
        )
        self.g3: sparray = csc_matrix(
            load_npz(Path.joinpath(simulation_path, submodel_config.inputs, "g3.npz"))
        )

        self._check_dimensions_spatial_matrices()

        # Load temporal_matrices
        self.m0: sparray = csc_matrix(
            load_npz(Path.joinpath(simulation_path, submodel_config.inputs, "m0.npz"))
        )
        self.m1: sparray = csc_matrix(
            load_npz(Path.joinpath(simulation_path, submodel_config.inputs, "m1.npz"))
        )
        self.m2: sparray = csc_matrix(
            load_npz(Path.joinpath(simulation_path, submodel_config.inputs, "m2.npz"))
        )

        self._check_dimensions_temporal_matrices()

        self.ns = self.c0.shape[0]  # Number of spatial nodes in the mesh
        self.nt = self.m0.shape[0]  # Number of temporal nodes in the mesh

        # Check that design_matrix shape match spatio-temporal fields
        assert (
            self.n_latent_parameters == self.ns * self.nt
        ), f"Design matrix has incorrect number of columns. \n    n_latent_parameters: {self.n_latent_parameters}\n    ns: {self.ns} * nt: {self.nt} = {self.ns * self.nt}"

        self.manifold = submodel_config.manifold

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

    def construct_Q_prior(self, **kwargs) -> sparray:
        """Construct the prior precision matrix."""

        theta: ArrayLike = kwargs.get("theta", None)
        theta_keys: list = kwargs.get("theta_keys", None)

        r_s = theta[theta_keys == "r_s"]
        r_t = theta[theta_keys == "r_t"]
        sigma_st = theta[theta_keys == "sigma_st"]

        gamma_s, gamma_t, gamma_st = self._interpretable2compute(
            r_s, r_t, sigma_st, dim_spatial_domain=2
        )

        exp_gamma_s = xp.exp(gamma_s)
        exp_gamma_t = xp.exp(gamma_t)
        exp_gamma_st = xp.exp(gamma_st)

        q1s = pow(exp_gamma_s, 2) * self.c0 + self.g1
        q2s = (
            pow(exp_gamma_s, 4) * self.c0 + 2 * pow(exp_gamma_s, 2) * self.g1 + self.g2
        )
        q3s = (
            pow(exp_gamma_s, 6) * self.c0
            + 3 * pow(exp_gamma_s, 4) * self.g1
            + 3 * pow(exp_gamma_s, 2) * self.g2
            + self.g3
        )

        # withsparseKroneckerProduct", color_id=0):
        Q_prior = csc_matrix(
            pow(exp_gamma_st, 2)
            * (
                kron(self.m0, q3s)
                + exp_gamma_t * kron(self.m1, q2s)
                + pow(exp_gamma_t, 2) * kron(self.m2, q1s)
            )
        )

        return Q_prior

    def _interpretable2compute(self, r_s, r_t, sigma_st, dim_spatial_domain=2) -> tuple:
        if dim_spatial_domain != 2:
            raise ValueError("Only 2D spatial domain is supported for now.")

        # Assumes alphas as fixed for now
        alpha_s = 2
        alpha_t = 1
        alpha_e = 1

        # implicit assumption that spatial domain is 2D
        alpha = alpha_e + alpha_s * (alpha_t - 0.5)

        nu_s = alpha - 1
        nu_t = alpha_t - 0.5

        gamma_s = 0.5 * xp.log(8 * nu_s) - r_s
        gamma_t = r_t - 0.5 * xp.log(8 * nu_t) + alpha_s * gamma_s

        if self.manifold == "sphere":
            cR_t = sp.special.gamma(nu_t) / (
                sp.special.gamma(alpha_t) * pow(4 * math.pi, 0.5)
            )
            c_s = 0.0
            for k in range(50):
                c_s += (2.0 * k + 1.0) / (
                    4.0 * math.pi * pow(pow(xp.exp(gamma_s), 2) + k * (k + 1), alpha)
                )
            gamma_st = 0.5 * xp.log(cR_t) + 0.5 * xp.log(c_s) - 0.5 * gamma_t - sigma_st

        elif self.manifold == "plane":
            c1_scaling_constant = pow(4 * math.pi, 1.5)
            c1 = (
                sp.special.gamma(nu_t)
                * sp.special.gamma(nu_s)
                / (
                    sp.special.gamma(alpha_t)
                    * sp.special.gamma(alpha)
                    * c1_scaling_constant
                )
            )
            gamma_st = 0.5 * xp.log(c1) - 0.5 * gamma_t - nu_s * gamma_s - sigma_st
        else:
            raise ValueError("Manifold not supported: ", self.manifold)

        return gamma_s, gamma_t, gamma_st

    def convert_theta_from_model2interpret(
        self, gamma_s, gamma_t, gamma_st, dim_spatial_domain=2
    ) -> tuple:
        """Convert theta from model scale to interpretable scale."""

        if dim_spatial_domain != 2:
            raise ValueError("Only 2D spatial domain is supported for now.")

        alpha_t = 1
        alpha_s = 2
        alpha_e = 1

        alpha = alpha_e + alpha_s * (alpha_t - 0.5)
        nu_s = alpha - 1
        nu_t = alpha_t - 0.5

        exp_gamma_s = xp.exp(gamma_s)
        exp_gamma_t = xp.exp(gamma_t)
        exp_gamma_st = xp.exp(gamma_st)

        r_s = xp.log(xp.sqrt(8 * nu_s) / exp_gamma_s)
        r_t = xp.log(exp_gamma_t * xp.sqrt(8 * nu_t) / (pow(exp_gamma_s, alpha_s)))

        if self.manifold == "sphere":
            c_r_t = sp.special.gamma(alpha_t - 0.5) / (
                sp.special.gamma(alpha_t) * pow(4 * math.pi, 0.5)
            )
            c_s = 0.0
            for k in range(50):
                c_s += (2.0 * k + 1) / (
                    4 * math.pi * pow(pow(exp_gamma_s, 2) + k * (k + 1), alpha)
                )
            sigma_st = xp.log(
                xp.sqrt(c_r_t * c_s) / (exp_gamma_st * xp.sqrt(exp_gamma_t))
            )

        elif self.manifold == "plane":
            c1_scaling_const = pow(4 * math.pi, dim_spatial_domain / 2.0) * pow(
                4 * math.pi, 0.5
            )
            c1 = (
                sp.special.gamma(nu_t)
                * sp.special.gamma(nu_s)
                / (
                    sp.special.gamma(alpha_t)
                    * sp.special.gamma(alpha)
                    * c1_scaling_const
                )
            )
            sigma_st = xp.log(
                xp.sqrt(c1)
                / (
                    (exp_gamma_st * xp.sqrt(exp_gamma_t))
                    * pow(exp_gamma_s, alpha - dim_spatial_domain / 2)
                )
            )

        else:
            raise ValueError("Manifold not supported: ", self.manifold)

        return r_s, r_t, sigma_st
