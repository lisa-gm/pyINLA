import pytest

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
