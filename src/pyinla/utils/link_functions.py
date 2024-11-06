# Copyright 2024 pyINLA authors. All rights reserved.

from autograd.numpy import exp

# import numpy as np


def sigmoid(x):
    return 1 / (1 + exp(-x))


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
