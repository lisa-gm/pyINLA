# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla.__init__ import xp


def dict2array(in_dict: dict) -> xp.ndarray:
    out_arr = xp.zeros(len(in_dict))
    for i, key in enumerate(in_dict.keys()):
        out_arr[i] = in_dict[key]

    return out_arr


# def theta_array2dict(
#     theta_array: np.ndarray, theta_model: dict, theta_likelihood: dict
# ) -> dict:
#     theta_model_values = theta_array[: len(theta_model)]
#     theta_likelihood_values = theta_array[len(theta_model) :]
#     theta_model = dict(zip(theta_model.keys(), theta_model_values))
#     theta_likelihood = dict(zip(theta_likelihood.keys(), theta_likelihood_values))
#     return theta_model, theta_likelihood
