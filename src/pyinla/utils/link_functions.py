# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla import xp


def sigmoid(x):
    return 1 / (1 + xp.exp(-x))
