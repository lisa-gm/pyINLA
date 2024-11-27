# Copyright 2024 pyINLA authors. All rights reserved.

import tomllib
from pathlib import Path
from typing import Literal
from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    model_validator,
    Field,
)
from typing_extensions import Self


# --- PRIOR HYPERPARAMETERS ----------------------------------------------------
class PriorHyperparametersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GaussianPriorHyperparametersConfig(PriorHyperparametersConfig):
    mean: float = 0.0
    precision: Annotated[float, Field(strict=True, gt=0)] = 0.5


class PenalizedComplexityPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: float = None
    u: float = None


# --- LIKELIHOODS --------------------------------------------------------------
class LikelihoodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gaussian", "poisson", "binomial"]


class GaussianLikelihoodConfig(LikelihoodConfig):
    prec_o: float = None  # Observation precision

    type: str = "gaussian"


class PoissonLikelihoodConfig(LikelihoodConfig):
    type: str = "poisson"


class BinomialLikelihoodConfig(LikelihoodConfig):
    type: str = "binomial"
    link_function: Literal["sigmoid"] = "sigmoid"


# --- SUBMODELS ----------------------------------------------------------------
class SubModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: str = None


class RegressionSubModelConfig(SubModelConfig):
    n_fixed_effects: Annotated[int, Field(strict=True, ge=1)] = 1
    fixed_effects_prior_precision: float = 0.001


class SpatioTemporalSubModelConfig(SubModelConfig):
    spatial_domain_dimension: PositiveInt = 2

    # --- Model hyperparameters in the interpretable scale ---
    r_s: float = None  # Spatial range
    r_t: float = None  # Temporal range
    sigma_st: float = None  # Spatio-temporal variation

    ph_s: PriorHyperparametersConfig = None
    ph_t: PriorHyperparametersConfig = None
    ph_st: PriorHyperparametersConfig = None


class SpatialSubModelConfig(SubModelConfig): ...


class TemporalSubModelConfig(SubModelConfig): ...


# --- MODEL --------------------------------------------------------------------
class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submodels: list[SubModelConfig] = None
    lh: LikelihoodConfig = None

    @model_validator(mode="after")
    def check_submodels(self) -> Self:
        assert self.submodels is not None, "At least one Submodels is required."
        assert all(
            isinstance(submodel, SubModelConfig) for submodel in self.submodels
        ), "All submodels must be instances of SubModelConfig."

        return self

    @model_validator(mode="after")
    def check_submodels_parameters(self) -> Self:
        for submodel in self.submodels:
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
        for submodel in self.submodels:
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


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["scipy", "serinv"] = "scipy"


class MinimizeConfig(BaseModel):
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
