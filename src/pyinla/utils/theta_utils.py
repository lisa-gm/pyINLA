# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np


def make_theta_array(theta_model, theta_likelihood):
    theta_model_values = list(theta_model.values())
    theta_likelihood_values = list(theta_likelihood.values())
    theta = np.array(theta_model_values + theta_likelihood_values)
    return theta
