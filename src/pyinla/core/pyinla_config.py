# Copyright 2024 pyINLA authors. All rights reserved.

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["regression", "spatio-temporal"] = None


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
