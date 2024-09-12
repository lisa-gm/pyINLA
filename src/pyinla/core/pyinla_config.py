# Copyright 2024 pyINLA authors. All rights reserved.

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, conint


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["regression", "spatio-temporal"] = None

    # --- Model parameters ---------------------------------------------
    n_fixed_effects: conint(ge=0)

    # Spatial model

    # Temporal model
    constraint_model: bool = False


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer: Literal["bfgs"] = "bfgs"


class PyinlaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Simulation parameters ---------------------------------------
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()

    # --- Directory paths ----------------------------------------------
    simulation_dir: Path = Path("./pyinla/")

    @property
    def input_dir(self) -> Path:
        return self.simulation_dir / "inputs/"


def parse_config(config_file: Path) -> PyinlaConfig:
    """Reads the TOML config file."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return PyinlaConfig(**config)
