# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
import pytest
from scipy import sparse

from pyinla import ArrayLike
from pyinla.core.likelihood import Likelihood
from pyinla.core.model import Model
from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.submodels.regression import RegressionModel
from pyinla.submodels.spatio_temporal import SpatioTemporalModel
from pyinla.utils import sigmoid


@pytest.fixture(scope="function", autouse=False)
def pyinla_config_initialize_theta(
    model: Model,
    likelihood: Likelihood,
    theta_prior_mean: ArrayLike,
    theta_prior_variance: ArrayLike,
):
    pyinla_config = PyinlaConfig()

    counter = 0
    if Model == RegressionModel and Likelihood == GaussianLikelihood:
        pyinla_config.prior_hyperparamters.mean_theta_observations = theta_prior_mean[
            counter
        ]
        pyinla_config.prior_hyperparamters.precision_theta_observations = (
            theta_prior_variance[counter]
        )
        counter += 1

    elif Model == SpatioTemporalModel:
        pyinla_config.prior_hyperparamters.mean_theta_spatial_range = theta_prior_mean[
            counter
        ]
        pyinla_config.prior_hyperparamters.precision_theta_spatial_range = (
            theta_prior_variance[counter]
        )
        counter += 1

        pyinla_config.prior_hyperparamters.mean_theta_temporal_range = theta_prior_mean[
            counter
        ]
        pyinla_config.prior_hyperparamters.precision_theta_temporal_range = (
            theta_prior_variance[counter]
        )
        counter += 1

        pyinla_config.prior_hyperparamters.mean_theta_spatio_temporal_variation = (
            theta_prior_mean[counter]
        )
        pyinla_config.prior_hyperparamters.precision_theta_spatio_temporal_variation = (
            theta_prior_variance[counter]
        )
        counter += 1

    else:
        print("undefined model!")
        raise ValueError

    return pyinla_config


N_OBSERVATIONS = [
    pytest.param(1, id="1_observation"),
    pytest.param(2, id="2_observations"),
    pytest.param(3, id="3_observations"),
]


N_LATENT_PARAMETERS = [
    pytest.param(1, id="1_latent_parameter"),
    pytest.param(2, id="2_latent_parameters"),
    pytest.param(3, id="3_latent_parameters"),
]


THETA_OBSERVATIONS = [
    pytest.param(-0.1, id="theta_observations_-0.1"),
    pytest.param(0.0, id="theta_observations_0.0"),
    pytest.param(0.1, id="theta_observations_0.1"),
]


@pytest.fixture(params=N_OBSERVATIONS, autouse=True)
def n_observations(request):
    return request.param


@pytest.fixture(params=N_LATENT_PARAMETERS, autouse=True)
def n_latent_parameters(request):
    return request.param


@pytest.fixture(params=THETA_OBSERVATIONS, autouse=True)
def theta_observations(request):
    return request.param


@pytest.fixture(scope="function", autouse=False)
def generate_gaussian_data(
    n_observations: int, n_latent_parameters: int, theta_observations: float
):
    theta_likelihood: dict = {"theta_observations": theta_observations}

    y = np.random.randn(n_observations)
    a = sparse.random(n_observations, n_latent_parameters, density=0.5)
    # generate x from a gaussian distribution of dimensions n_latent_parameters with mean 0 and precision exp(theta_observations)
    variance = 1 / np.exp(theta_observations)
    x = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=n_latent_parameters)

    return a, x, y, theta_likelihood


@pytest.fixture(scope="function", autouse=False)
def generate_poisson_data(
    n_observations: int, n_latent_parameters: int, theta_observations: float
):
    theta_likelihood: dict = {"theta_observations": theta_observations}

    a = sparse.random(n_observations, n_latent_parameters, density=0.5)
    variance = 1 / np.exp(theta_observations)
    x = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=n_latent_parameters)
    eta = a @ x
    lam = np.exp(eta)
    y = np.random.poisson(lam=lam)

    return eta, y, lam, theta_likelihood


@pytest.fixture(scope="function", autouse=False)
def generate_binomial_data(
    n_observations: int,
    n_latent_parameters: int,
    theta_observations: float,
):
    theta_likelihood: dict = {"theta_observations": theta_observations}

    a = sparse.random(n_observations, n_latent_parameters, density=0.5)
    x = np.random.randn(n_latent_parameters)
    eta = a @ x
    # n_trials = np.ones(len(eta), dtype=int)
    n_trials = np.random.randint(1, 11, size=n_observations)
    prob = sigmoid(eta)
    y = np.random.binomial(n=n_trials, p=prob)

    return eta, y, n_trials, prob, theta_likelihood
