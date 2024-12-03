# Copyright 2024 pyINLA authors. All rights reserved.

import tomllib
from pathlib import Path
from typing import Literal
from typing_extensions import Annotated

from abc import ABC, abstractmethod

from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    model_validator,
    Field,
)
from typing_extensions import Self

from pyinla.__init__ import xp, ArrayLike


# --- PRIOR HYPERPARAMETERS ----------------------------------------------------
class PriorHyperparametersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GaussianPriorHyperparametersConfig(PriorHyperparametersConfig):
    mean: float = 0.0
    precision: Annotated[float, Field(strict=True, gt=0)] = 0.5


class PenalizedComplexityPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: float = None
    u: float = None

    # Generalized formula:
    # lambda = - log(alpha) * pow(u, c_l)
    #
    # log_prior = a + b + c
    # a = log(lambda)
    # b = -lambda * exp(c_b * r)
    # c = c_c * r


# --- LIKELIHOODS --------------------------------------------------------------
class LikelihoodConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gaussian", "poisson", "binomial"]

    prior_hyperparameters: PriorHyperparametersConfig = None

    @abstractmethod
    def read_hyperparameters(self) -> tuple[ArrayLike, list]: ...


class GaussianLikelihoodConfig(LikelihoodConfig):
    prec_o: float = None  # Observation precision

    type: str = "gaussian"

    def read_hyperparameters(self):
        theta = xp.array([self.prec_o])
        theta_keys = ["prec_o"]

        return theta, theta_keys


class PoissonLikelihoodConfig(LikelihoodConfig):
    type: str = "poisson"

    def read_hyperparameters(self):
        return [], []


class BinomialLikelihoodConfig(LikelihoodConfig):
    type: str = "binomial"
    link_function: Literal["sigmoid"] = "sigmoid"

    def read_hyperparameters(self):
        return [], []


# --- SUBMODELS ----------------------------------------------------------------
class SubModelConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    # Input folder for this specific submodel
    inputs: str = None

    @abstractmethod
    def read_hyperparameters(self) -> tuple[ArrayLike, list]: ...


class RegressionSubModelConfig(SubModelConfig):
    n_fixed_effects: Annotated[int, Field(strict=True, ge=1)] = 1
    fixed_effects_prior_precision: float = 0.001

    def read_hyperparameters(self):
        return [], []


class SpatioTemporalSubModelConfig(SubModelConfig):
    spatial_domain_dimension: PositiveInt = 2

    # --- Model hyperparameters in the interpretable scale ---
    r_s: float = None  # Spatial range
    r_t: float = None  # Temporal range
    sigma_st: float = None  # Spatio-temporal variation

    ph_s: PriorHyperparametersConfig = None
    ph_t: PriorHyperparametersConfig = None
    ph_st: PriorHyperparametersConfig = None

    manifold: Literal["plane", "sphere"] = "plane"

    def read_hyperparameters(self):
        theta = xp.array([self.r_s, self.r_t, self.sigma_st])
        theta_keys = ["r_s", "r_t", "sigma_st"]

        return theta, theta_keys


class SpatialSubModelConfig(SubModelConfig): ...


class TemporalSubModelConfig(SubModelConfig): ...


# --- MODEL --------------------------------------------------------------------
class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submodels: list[SubModelConfig] = None
    likelihood: LikelihoodConfig = None


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["dense", "scipy", "serinv"] = "scipy"


class MinimizeConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    max_iter: PositiveInt = 100
    jac: bool = True


class BFGSConfig(MinimizeConfig):
    gtol: float = 1e-1
    c1: float = None
    c2: float = None
    disp: bool = False


class PyinlaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Model parameters -----------------------------------------------------
    model: ModelConfig = ModelConfig()

    @model_validator(mode="after")
    def check_submodels(self) -> Self:
        assert self.model.submodels is not None, "At least one Submodels is required."
        assert all(
            isinstance(submodel, SubModelConfig) for submodel in self.model.submodels
        ), "All submodels must be instances of SubModelConfig."
        return self

    @model_validator(mode="after")
    def check_likelihood(self) -> Self:
        assert self.model.likelihood is not None, "Likelihood is required."
        assert isinstance(
            self.model.likelihood, LikelihoodConfig
        ), "Likelihood must be an instance of LikelihoodConfig."
        return self

    @model_validator(mode="after")
    def check_single_spatio_temporal_submodel(self) -> Self:
        n_spatio_temporal_submodels = sum(
            isinstance(submodel, SpatioTemporalSubModelConfig)
            for submodel in self.model.submodels
        )
        assert (
            n_spatio_temporal_submodels == 1
        ), "Only one SpatioTemporalSubModel is allowed."
        return self

    @model_validator(mode="after")
    def check_submodels_parameters(self) -> Self:
        for submodel in self.model.submodels:
            assert submodel.inputs is not None, "Submodel input folder is required."

            # Submodel specific checks
            if isinstance(submodel, RegressionSubModelConfig):
                ...

            if isinstance(submodel, SpatioTemporalSubModelConfig):
                assert (
                    submodel.r_s is not None
                ), "Spatial range is required for SpatioTemporalSubModel."
                assert (
                    submodel.r_t is not None
                ), "Temporal range is required for SpatioTemporalSubModel."
                assert (
                    submodel.sigma_st is not None
                ), "Spatio-temporal variation is required for SpatioTemporalSubModel."
        return self

    @model_validator(mode="after")
    def check_priorhyperparameters(self) -> Self:
        for submodel in self.model.submodels:
            if isinstance(submodel, RegressionSubModelConfig):
                # Regression model does not have prior hyperparameters
                ...
            if isinstance(submodel, SpatioTemporalSubModelConfig):
                assert (
                    submodel.ph_s is not None
                ), "Spatial prior hyperparameters are required for SpatioTemporalSubModel."
                if isinstance(
                    submodel.ph_s, PenalizedComplexityPriorHyperparametersConfig
                ):
                    assert (
                        submodel.ph_s.alpha is not None
                    ), "Alpha is required for PenalizedComplexityPriorHyperparametersConfig."
                    assert (
                        submodel.ph_s.u is not None
                    ), "U is required for PenalizedComplexityPriorHyperparametersConfig."

                assert (
                    submodel.ph_t is not None
                ), "Temporal prior hyperparameters are required for SpatioTemporalSubModel."
                if isinstance(
                    submodel.ph_t, PenalizedComplexityPriorHyperparametersConfig
                ):
                    assert (
                        submodel.ph_t.alpha is not None
                    ), "Alpha is required for PenalizedComplexityPriorHyperparametersConfig."
                    assert (
                        submodel.ph_t.u is not None
                    ), "U is required for PenalizedComplexityPriorHyperparametersConfig."

                assert (
                    submodel.ph_st is not None
                ), "Spatio-temporal prior hyperparameters are required for SpatioTemporalSubModel."
                if isinstance(
                    submodel.ph_st, PenalizedComplexityPriorHyperparametersConfig
                ):
                    assert (
                        submodel.ph_st.alpha is not None
                    ), "Alpha is required for PenalizedComplexityPriorHyperparametersConfig."
                    assert (
                        submodel.ph_st.u is not None
                    ), "U is required for PenalizedComplexityPriorHyperparametersConfig."
        return self

    # --- Simulation parameters ------------------------------------------------
    solver: SolverConfig = SolverConfig()
    minimize: MinimizeConfig = BFGSConfig()

    inner_iteration_max_iter: PositiveInt = 50
    eps_inner_iteration: float = 1e-3
    eps_gradient_f: float = 1e-3

    # --- Directory paths ------------------------------------------------------
    simulation_dir: Path = Path("./pyinla/")
    input_dir: Path = Path.joinpath(simulation_dir, "inputs/")

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
