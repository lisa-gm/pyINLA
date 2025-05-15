# Copyright 2024-2025 pyINLA authors. All rights reserved.

import tomllib
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from typing_extensions import Annotated

from pyinla.__init__ import ArrayLike, NDArray, xp
from pyinla.configs.priorhyperparameters_config import PriorHyperparametersConfig, BetaPriorHyperparametersConfig, GaussianMVNPriorHyperparametersConfig
from pyinla.configs.priorhyperparameters_config import (
    parse_config as parse_priorhyperparameters_config,
)
from pyinla.utils import scaled_logit


class SubModelConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Input folder for this specific submodel
    input_dir: str = None
    type: Literal["spatio_temporal", "spatial", "regression", "brainiac"] = None

    @abstractmethod
    def read_hyperparameters(self) -> tuple[ArrayLike, list]: ...


class RegressionSubModelConfig(SubModelConfig):
    n_fixed_effects: Annotated[int, Field(strict=True, ge=1)] = 1
    fixed_effects_prior_precision: float = 0.001

    def read_hyperparameters(self):
        return xp.array([]), []


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


class SpatialSubModelConfig(SubModelConfig):
    spatial_domain_dimension: PositiveInt = 2

    # --- Model hyperparameters in the interpretable scale ---
    r_s: float = None  # Spatial range
    sigma_e: float = None  # Spatial variation

    ph_s: PriorHyperparametersConfig = None
    ph_e: PriorHyperparametersConfig = None

    def read_hyperparameters(self):
        theta = xp.array([self.r_s, self.sigma_e])
        theta_keys = ["r_s", "sigma_e"]

        return theta, theta_keys


class TemporalSubModelConfig(SubModelConfig): ...


class BrainiacSubModelConfig(SubModelConfig):
    # --- Hyperparameters ---
    h2: float = None
    h2_scaled: float = None
    alpha: NDArray = None 

    # --- Prior hyperparameters ---
    ph_h2: BetaPriorHyperparametersConfig = None
    ph_alpha: GaussianMVNPriorHyperparametersConfig = None

    def read_hyperparameters(self):

        # input of h2 is in (0,1), rescale to -/+ INF
        self.h2_scaled = scaled_logit(self.h2, direction="forward")

        theta = xp.concatenate(([self.h2_scaled], self.alpha))
        theta_keys = ["h2"] + [f"alpha_{i}" for i in range(len(self.alpha))]

        return theta, theta_keys


def parse_config(config: dict | str) -> SubModelConfig:
    if isinstance(config, str):
        with open(config, "rb") as f:
            config = tomllib.load(f)

    type = config.get("type")
    if type == "spatio_temporal":
        config["ph_s"] = parse_priorhyperparameters_config(config["ph_s"])
        config["ph_t"] = parse_priorhyperparameters_config(config["ph_t"])
        config["ph_st"] = parse_priorhyperparameters_config(config["ph_st"])
        return SpatioTemporalSubModelConfig(**config)
    elif type == "spatial":
        config["ph_s"] = parse_priorhyperparameters_config(config["ph_s"])
        config["ph_e"] = parse_priorhyperparameters_config(config["ph_e"])
        return SpatialSubModelConfig(**config)
    elif type == "regression":
        return RegressionSubModelConfig(**config)
    elif type == "brainiac":
        config["ph_h2"] = parse_priorhyperparameters_config(config["ph_h2"])
        config["ph_alpha"] = parse_priorhyperparameters_config(config["ph_alpha"])
        return BrainiacSubModelConfig(**config)
    # Add more elif branches for other submodel types
    else:
        raise ValueError(f"Unknown submodel type: {type}")
