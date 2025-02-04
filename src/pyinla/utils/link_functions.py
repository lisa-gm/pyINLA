# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import NDArray, xp


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + xp.exp(-x))
