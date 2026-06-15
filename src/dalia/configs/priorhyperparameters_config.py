# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat
from scipy.sparse import spmatrix
from typing_extensions import Annotated

from dalia.__init__ import NDArray


# --- PRIOR HYPERPARAMETERS ----------------------------------------------------
class PriorHyperparametersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    type: Literal[
        "gaussian",
        "penalized_complexity",
        "beta",
        "gaussian_mvn",
        "gamma",
        "inverse_gamma",
        "half_cauchy",
        "half_normal",
    ] = None


class GaussianPriorHyperparametersConfig(PriorHyperparametersConfig):
    mean: float = 0.0
    precision: Annotated[float, Field(strict=True, gt=0)] = 0.5


class GaussianMVNPriorHyperparametersConfig(PriorHyperparametersConfig):
    mean: NDArray = None
    precision: spmatrix = None


class PenalizedComplexityPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: float = None
    u: float = None

class BetaPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: PositiveFloat = None
    beta: PositiveFloat = None


class GammaPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: PositiveFloat = None
    beta: PositiveFloat = None

class InverseGammaPriorHyperparametersConfig(PriorHyperparametersConfig):
    alpha: PositiveFloat = None
    beta: PositiveFloat = None


class HalfCauchyPriorHyperparametersConfig(PriorHyperparametersConfig):
    scale: PositiveFloat = 25.0


class HalfNormalPriorHyperparametersConfig(PriorHyperparametersConfig):
    precision: PositiveFloat = 0.001


def parse_config(config: dict) -> PriorHyperparametersConfig:
    prior_type = config.get("type")
    if prior_type == "gaussian":
        return GaussianPriorHyperparametersConfig(**config)
    if prior_type == "gaussian_mvn":
        return GaussianMVNPriorHyperparametersConfig(**config)
    if prior_type == "penalized_complexity":
        return PenalizedComplexityPriorHyperparametersConfig(**config)
    if prior_type == "beta":
        return BetaPriorHyperparametersConfig(**config)
    if prior_type == "gamma":
        return GammaPriorHyperparametersConfig(**config)
    if prior_type == "inverse_gamma":
        return InverseGammaPriorHyperparametersConfig(**config)
    if prior_type == "half_cauchy":
        return HalfCauchyPriorHyperparametersConfig(**config)
    if prior_type == "half_normal":
        return HalfNormalPriorHyperparametersConfig(**config)
    raise ValueError(f"Unknown prior hyperparameters config type: {prior_type}")
