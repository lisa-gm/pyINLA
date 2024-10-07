# Copyright 2024 pyINLA authors. All rights reserved.

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt, conint, model_validator
from typing_extensions import Self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["regression", "spatio-temporal"] = None

    # --- Model parameters -----------------------------------------------------
    n_fixed_effects: conint(ge=0) = 0
    fixed_effects_prior_precision: float = 0.001

    # ----- Regression model -----
    ...

    # ----- Spatio-temporal model -----
    constraint_model: bool = False

    spatial_domain_dimension: PositiveInt = 2

    theta_spatial_range: float = None
    theta_temporal_range: float = None
    theta_spatio_temporal_variation: float = None

    @model_validator(mode="after")
    def check_theta(self) -> Self:
        if self.type == "spatio-temporal":
            assert self.theta_spatial_range is not None, "Spatial range is required."
            assert self.theta_temporal_range is not None, "Temporal range is required."
            assert (
                self.theta_spatio_temporal_variation is not None
            ), "Spatio-temporal variation is required."

        return self


class PriorHyperparametersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gaussian", "penalized-complexity"] = "gaussian"

    # --- Gaussian prior hyperparameters ---------------------------------------
    # ----- Models -----
    # Regression
    ...

    # Spatio-temporal
    mean_theta_spatial_range: float = 0.0
    mean_theta_temporal_range: float = 0.0
    mean_theta_spatio_temporal_variation: float = 0.0

    precision_theta_spatial_range: float = 0.5
    precision_theta_temporal_range: float = 0.5
    precision_theta_spatio_temporal_variation: float = 0.5

    # ----- Likelihood -----
    # Gaussian likelihood
    mean_theta_observations: float = 0.0
    precision_theta_observations: float = 0.5

    # Poisson likelihood
    ...

    # Binomial likelihood
    ...

    # --- Penalized complexity prior hyperparameters ---------------------------
    # ----- Models -----
    # Regression
    ...

    # Spatio-temporal
    alpha_theta_spatial_range: float = None
    alpha_theta_temporal_range: float = None
    alpha_theta_spatio_temporal_variation: float = None

    u_theta_spatial_range: float = None
    u_theta_temporal_range: float = None
    u_theta_spatio_temporal_variation: float = None

    @model_validator(mode="after")
    def check_alpha(self) -> Self:
        if self.type == "penalized-complexity":
            assert (
                self.alpha_theta_spatial_range is not None
            ), "Spatial range alpha is required."
            assert (
                self.alpha_theta_temporal_range is not None
            ), "Temporal range alpha is required."
            assert (
                self.alpha_theta_spatio_temporal_variation is not None
            ), "Spatio-temporal standard deviation alpha is required."

        return self

    @model_validator(mode="after")
    def check_u(self) -> Self:
        if self.type == "penalized-complexity":
            assert (
                self.u_theta_spatial_range is not None
            ), "Spatial range u is required."
            assert (
                self.u_theta_temporal_range is not None
            ), "Temporal range u is required."
            assert (
                self.u_theta_spatio_temporal_variation is not None
            ), "Spatio-temporal standard deviation u is required."

        return self

    # ----- Likelihood -----
    # Gaussian likelihood
    alpha_theta_observations: float = None
    u_theta_observations: float = None

    # Poisson likelihood
    ...

    # Binomial likelihood
    ...


class LikelihoodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gaussian", "poisson", "binomial"] = None

    # --- Gaussian likelihood --------------------------------------------------
    theta_observations: float = None

    @model_validator(mode="after")
    def check_theta(self) -> Self:
        if self.type == "gaussian":
            assert (
                self.theta_observations is not None
            ), "In case of a Gaussian likelihood the observations hyperparameter is required."

        return self

    # --- Poisson likelihood ---------------------------------------------------
    ...

    # --- Binomial likelihood --------------------------------------------------
    link_function: Literal["sigmoid"] = "sigmoid"


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["scipy", "serinv"] = "scipy"


class PyinlaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Model parameters -----------------------------------------------------
    model: ModelConfig = ModelConfig()
    prior_hyperparameters: PriorHyperparametersConfig = PriorHyperparametersConfig()
    likelihood: LikelihoodConfig = LikelihoodConfig()

    # --- Simulation parameters ------------------------------------------------
    solver: SolverConfig = SolverConfig()
    minimize_max_iter: PositiveInt = 100
    eps_inner_iteration: float = 1e-3
    eps_gradient_f: float = 1e-3

    # --- Directory paths ------------------------------------------------------
    simulation_dir: Path = Path("./pyinla/")

    @property
    def input_dir(self) -> Path:
        return self.simulation_dir / "inputs/"

    @model_validator(mode="after")
    def check_likelihood_prior_hyperparameters(self) -> Self:
        if self.prior_hyperparameters.type == "penalized-complexity":
            if self.likelihood.type == "gaussian":
                assert (
                    self.prior_hyperparameters.alpha_theta_observations is not None
                ), "Gaussian likelihood requires alpha theta observations."
                assert (
                    self.prior_hyperparameters.u_theta_observations is not None
                ), "Gaussian likelihood requires u theta observations."
        return self


def parse_config(config_file: Path) -> PyinlaConfig:
    """Reads the TOML config file."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return PyinlaConfig(**config)
