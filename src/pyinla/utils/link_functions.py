# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla import xp, NDArray


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + xp.exp(-x))
