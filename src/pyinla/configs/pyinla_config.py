# Copyright 2024 pyINLA authors. All rights reserved.

import tomllib
from abc import ABC
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt


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

    # --- Simulation parameters ------------------------------------------------
    solver: SolverConfig = SolverConfig()
    minimize: MinimizeConfig = BFGSConfig()

    inner_iteration_max_iter: PositiveInt = 50
    eps_inner_iteration: float = 1e-3
    eps_gradient_f: float = 1e-3

    # --- Directory paths ------------------------------------------------------
    simulation_dir: Path = Path("./pyinla/")
    output_dir: Path = Path.joinpath(simulation_dir, "output/")


def parse_config(config_file: Path) -> PyinlaConfig:
    """Reads the TOML config file."""
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    return PyinlaConfig(**config)
