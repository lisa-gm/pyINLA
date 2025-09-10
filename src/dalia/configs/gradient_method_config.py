from abc import ABC
from typing import Literal

from pydantic import BaseModel, ConfigDict


class GradientMethodConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Input folder for this specific submodel
    input_dir: str = None
    type: Literal["vanilla_gradient", "smart_gradient"] = "smart_gradient"

    finite_difference_epsilon: float = 1e-3

class VanillaGradientConfig(GradientMethodConfig): ...

class SmartGradientConfig(GradientMethodConfig):
    # The diagonal noise ensure to avoid singularities in the QR decomposition
    diagonal_noise: float = 1e-8
    # Threshold below which scaling is not performed
    scaling_threashold: float = 1e-12