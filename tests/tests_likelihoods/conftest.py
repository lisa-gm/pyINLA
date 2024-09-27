import numpy as np
import pytest
from scipy import sparse

N_OBSERVATIONS = [1, 2, 3, 9]
N_LATENT_PARAMETERS = [1, 2, 3, 8]
THETA_OBSERVATIONS = [-0.1, 0.0, 0.1, 0.2]


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
