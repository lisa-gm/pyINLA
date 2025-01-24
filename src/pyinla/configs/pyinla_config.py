# Copyright 2024 pyINLA authors. All rights reserved.

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["dense", "scipy", "serinv"] = "scipy"


class BFGSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iter: PositiveInt = 100
    jac: bool = True

    gtol: float = 1e-1
    c1: float = 1e-4 # Default value from the scipy documentation
    c2: float = 0.9 # Default value from the scipy documentation
    disp: bool = False


class PyinlaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Simulation parameters ------------------------------------------------
    solver: SolverConfig = SolverConfig()
    minimize: BFGSConfig = BFGSConfig()

    inner_iteration_max_iter: PositiveInt = 50
    eps_inner_iteration: float = 1e-3
    eps_gradient_f: float = 1e-3

    # --- Directory paths ------------------------------------------------------
    simulation_dir: Path = Path("./pyinla/")
    output_dir: Path = Path.joinpath(simulation_dir, "output/")


def parse_config(config: dict | str) -> PyinlaConfig:
    if isinstance(config, str):
        with open(config, "rb") as f:
            config = tomllib.load(f)

    return PyinlaConfig(**config)
