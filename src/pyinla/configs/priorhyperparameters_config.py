# Copyright 2024-2025 pyINLA authors. All rights reserved.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from pyinla.__init__ import NDArray
from scipy.sparse import spmatrix

# --- PRIOR HYPERPARAMETERS ----------------------------------------------------
class PriorHyperparametersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    type: Literal["gaussian", "penalized_complexity", "beta", "gaussian_mvn"] = None


class GaussianPriorHyperparametersConfig(PriorHyperparametersConfig):
    mean: float = 0.0
    precision: Annotated[float, Field(strict=True, gt=0)] = 0.5


class GaussianMVNPriorHyperparametersConfig(PriorHyperparametersConfig):
    mean: NDArray = None
    precision: spmatrix = None


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


class BetaPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: float = None
    beta: float = None


def parse_config(config: dict) -> PriorHyperparametersConfig:
    prior_type = config.get("type")
    if prior_type == "gaussian":
        return GaussianPriorHyperparametersConfig(**config)
    elif prior_type == "gaussian_mvn":
        return GaussianMVNPriorHyperparametersConfig(**config)
    elif prior_type == "penalized_complexity":
        return PenalizedComplexityPriorHyperparametersConfig(**config)
    elif prior_type == "beta":
        return BetaPriorHyperparametersConfig(**config)
    else:
        raise ValueError(f"Unknown prior hyperparameters config type: {prior_type}")
