# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
