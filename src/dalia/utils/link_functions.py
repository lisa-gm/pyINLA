# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import NDArray, xp


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + xp.exp(-x))


def cloglog(x: NDArray, direction: str) -> NDArray:
    if direction == "forward":
        return xp.log(-xp.log(1 - x))
    elif direction == "backward":
        return 1 - xp.exp(-xp.exp(x))
    else:
        raise ValueError(f"Unknown direction: {direction}")


def scaled_logit(x: NDArray, direction: str) -> NDArray:
    k = 1.0 / 12.0
    if direction == "forward":
        ## TODO: check special function log1p
        # input must be between 0 and 1
        if xp.min(x) <= 0 or xp.max(x) >= 1:
            raise ValueError(
                f"Input to scaled_logit forward must be in (0,1). Got x={x}"
            )
        return (1.0 / k) * xp.log(x / (1.0 - x))
    elif direction == "backward":
        return 1 / (1 + xp.exp(-k * x))
    elif direction == "forward_jacobian":
        if xp.min(x) <= 0 or xp.max(x) >= 1:
            raise ValueError(
                f"Input to scaled_logit forward_jacobian must be in (0,1). Got x={x}"
            )
        return 1.0 / (k * x * (1.0 - x))
    elif direction == "backward_log_jacobian":
        return xp.log(k) - 2 * xp.log1p(xp.exp(-k * x)) - k * x
    else:
        raise ValueError(f"Unknown direction: {direction}")
