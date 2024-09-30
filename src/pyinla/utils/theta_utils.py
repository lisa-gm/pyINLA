# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np


def theta_dict2array(theta_model: dict, theta_likelihood: dict) -> np.ndarray:
    theta_model_values = list(theta_model.values())
    theta_likelihood_values = list(theta_likelihood.values())
    theta_array = np.concatenate((theta_model_values, theta_likelihood_values))
    return theta_array


def theta_array2dict(
    theta_array: np.ndarray, theta_model: dict, theta_likelihood: dict
) -> dict:
    theta_model_values = theta_array[: len(theta_model)]
    theta_likelihood_values = theta_array[len(theta_model) :]
    theta_model = dict(zip(theta_model.keys(), theta_model_values))
    theta_likelihood = dict(zip(theta_likelihood.keys(), theta_likelihood_values))
    return theta_model, theta_likelihood
