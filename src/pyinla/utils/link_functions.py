# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import NDArray, xp


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + xp.exp(-x))


def cloglog(x: NDArray, direction: str) -> NDArray:
    if direction == "forward":
        return xp.log(-xp.log(1 - x))
    elif direction == "backward":
        return 1 - xp.exp(-xp.exp(x))
    else:
        raise ValueError(f"Unknown direction: {direction}")
