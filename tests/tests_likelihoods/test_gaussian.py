import numpy as np
import pytest
from scipy import sparse
from scipy.stats import multivariate_normal

from pyinla.core.likelihood import Likelihood
from pyinla.likelihoods.gaussian import GaussianLikelihood


@pytest.mark.parametrize("likelihood", [GaussianLikelihood])  #
@pytest.mark.parametrize("n_observations", [1, 2, 3, 9])  #
@pytest.mark.parametrize("n_latent_parameters", [1, 2, 3, 8])  #
@pytest.mark.parametrize("theta_observations", [-0.1, 0.0, 0.1, 0.2])  #
def test_gaussian(
    likelihood: Likelihood,
    n_observations: int,
    n_latent_parameters: int,
    theta_likelihood: dict,
    theta_observations: float,
    pyinla_config,
):
    theta_likelihood["theta_observations"] = theta_observations

    y = np.random.randn(n_observations)
    a = sparse.random(n_observations, n_latent_parameters, density=0.5)
    x = np.random.randn(n_latent_parameters)
    eta = a @ x

    likelihood_instance = likelihood(pyinla_config, n_observations)

    likelihood_inla = likelihood_instance.evaluate_likelihood(y, eta, theta_likelihood)
    # print(f"likelihood_inla: {likelihood_inla}")

    # reference using scipy.stats.multivariate_normal
    Sigma = 1 / np.exp(theta_observations) * np.eye(n_observations)
    multivariate_normal_ref = multivariate_normal.logpdf(y, mean=a @ x, cov=Sigma)
    # deduct constant
    multivariate_normal_ref += 0.5 * n_observations * np.log(2 * np.pi)
    # print(f"multivariate_normal_ref w/o constant: {multivariate_normal_ref}")

    assert np.allclose(likelihood_inla, multivariate_normal_ref)
