# Copyright 2024-2025 pyINLA authors. All rights reserved.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


# --- PRIOR HYPERPARAMETERS ----------------------------------------------------
class PriorHyperparametersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gaussian", "penalized_complexity"] = None


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


def parse_config(config: dict) -> PriorHyperparametersConfig:
    prior_type = config.get("type")
    if prior_type == "gaussian":
        return GaussianPriorHyperparametersConfig(**config)
    elif prior_type == "penalized_complexity":
        return PenalizedComplexityPriorHyperparametersConfig(**config)
    else:
        raise ValueError(f"Unknown prior hyperparameters config type: {prior_type}")
