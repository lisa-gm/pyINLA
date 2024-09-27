import pytest

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
