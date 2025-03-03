# Copyright 2024-2025 pyINLA authors. All rights reserved.

import tomllib
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from typing_extensions import Annotated

from pyinla.__init__ import ArrayLike, xp
from pyinla.configs.priorhyperparameters_config import PriorHyperparametersConfig
from pyinla.configs.priorhyperparameters_config import (
    parse_config as parse_priorhyperparameters_config,
)


class SubModelConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    # Input folder for this specific submodel
    input_dir: str = None
    type: Literal["spatio_temporal", "spatial", "regression"] = None

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


""" class CoregionalizationSubModelConfig(SubModelConfig):

    submodel_type: Literal["spatial", "spatio-temporal"] = None
    num_vars: PositiveInt = 2

    if num_vars != 2 or num_vars != 3:
        raise ValueError("Invalid number of variables. Must be 1,2 or 3.")

    submodel_config_list = []
    if submodel_type == "spatial":
        for i in range(num_vars):
            submodel_config_list.append(SpatialSubModelConfig)

    elif submodel_type == "spatio-temporal":
        for i in range(num_vars):
            submodel_config_list.append(SpatioTemporalSubModelConfig)

    # import quantities from submodels

    # scaling parameter
    sigma_z1: float = None
    sigma_z2: float = None
    lambda1: float = None

    ph_sigma_z1: PriorHyperparametersConfig = None
    ph_sigma_z2: PriorHyperparametersConfig = None
    ph_lambda1: PriorHyperparametersConfig = None

    if num_vars == 3:
        sigma_z3: float = None
        lambda2: float = None
        lambda3: float = None

        ph_sigma_z3: PriorHyperparametersConfig = None
        ph_lambda2: PriorHyperparametersConfig = None
        ph_lambda3: PriorHyperparametersConfig = None

    def read_hyperparameters(self):

        for i in range(self.num_vars):

            theta_1, theta_keys_1 = self.submodel_config_1.read_hyperparameters()
            theta_2, theta_keys_2 = self.submodel_config_2.read_hyperparameters()

        theta = xp.concatenate([theta_1, theta_2])
        theta_keys = theta_keys_1 + theta_keys_2

        return theta, theta_keys """


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
    # Add more elif branches for other submodel types
    else:
        raise ValueError(f"Unknown submodel type: {type}")
