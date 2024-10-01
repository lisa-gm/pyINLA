# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy import sparse

from pyinla.core.likelihood import Likelihood
from pyinla.core.model import Model
from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.models.regression import RegressionModel
from pyinla.models.spatio_temporal import SpatioTemporalModel
from pyinla.solvers.scipy_solver import ScipySolver
from pyinla.utils import sigmoid

SOLVER = [ScipySolver]


@pytest.fixture(params=SOLVER, autouse=True)
def solver(request):
    return request.param


@pytest.fixture(scope="function", autouse=False)
def pobta_dense(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
):
    """Returns a random, positive definite, block tridiagonal arrowhead matrix."""

    pobta_dense = np.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=np.float64,
    )

    if arrowhead_blocksize != 0:
        # Fill the lower arrowhead blocks
        pobta_dense[-arrowhead_blocksize:, :-arrowhead_blocksize] = np.random.rand(
            arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
        )

        # Fill the tip of the arrowhead
        pobta_dense[-arrowhead_blocksize:, -arrowhead_blocksize:] = np.random.rand(
            arrowhead_blocksize, arrowhead_blocksize
        )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        pobta_dense[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = np.random.rand(diagonal_blocksize, diagonal_blocksize) + np.eye(
            diagonal_blocksize
        )

        # Fill the off-diagonal blocks
        if i > 0:
            pobta_dense[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ] = np.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make the matrix diagonally dominant
    for i in range(pobta_dense.shape[0]):
        pobta_dense[i, i] = 1 + np.sum(pobta_dense[i, :])

    # Make the matrix symmetric
    pobta_dense = (pobta_dense + pobta_dense.T) / 2

    return pobta_dense


@pytest.fixture(scope="function", autouse=False)
def pyinla_config(solver):
    """Returns a PyinlaConfig object."""

    pyinla_config = PyinlaConfig()

    if solver == "ScipySolver":
        pyinla_config.solver.type = "scipy"
    elif solver == "SerinvSolver":
        pyinla_config.solver.type = "serinv"

    return pyinla_config


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
    pytest.param(9, id="9_observations"),
]


N_LATENT_PARAMETERS = [
    pytest.param(1, id="1_latent_parameter"),
    pytest.param(2, id="2_latent_parameters"),
    pytest.param(3, id="3_latent_parameters"),
    pytest.param(8, id="8_latent_parameters"),
]


THETA_OBSERVATIONS = [
    pytest.param(-0.1, id="theta_observations_-0.1"),
    pytest.param(0.0, id="theta_observations_0.0"),
    pytest.param(0.1, id="theta_observations_0.1"),
    pytest.param(0.2, id="theta_observations_0.2"),
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

    return y, eta, lam, theta_likelihood


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
    n_trials = np.ones(len(eta), dtype=int)
    prob = sigmoid(eta)
    y = np.random.binomial(n=n_trials, p=prob)

    return y, eta, n_trials, prob, theta_likelihood
