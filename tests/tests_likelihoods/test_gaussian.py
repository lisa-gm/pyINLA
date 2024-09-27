import numpy as np
from scipy import sparse
from scipy.stats import multivariate_normal

from pyinla.likelihoods.gaussian import GaussianLikelihood


def test_gaussian_evaluate_likelihood(
    n_observations: int,
    n_latent_parameters: int,
    theta_observations: float,
    pyinla_config,
):
    theta_likelihood: dict = {"theta_observations": theta_observations}

    y = np.random.randn(n_observations)
    a = sparse.random(n_observations, n_latent_parameters, density=0.5)
    x = np.random.randn(n_latent_parameters)
    eta = a @ x

    likelihood_instance = GaussianLikelihood(pyinla_config, n_observations)

    likelihood_inla = likelihood_instance.evaluate_likelihood(y, eta, theta_likelihood)

    # Reference using scipy.stats.multivariate_normal
    Sigma = 1 / np.exp(theta_observations) * np.eye(n_observations)
    multivariate_normal_ref = multivariate_normal.logpdf(y, mean=a @ x, cov=Sigma)
    # Deduct constant
    multivariate_normal_ref += 0.5 * n_observations * np.log(2 * np.pi)

    assert np.allclose(likelihood_inla, multivariate_normal_ref)


def test_gaussian_evaluate_gradient(): ...


def test_gaussian_evaluate_hessian(): ...
