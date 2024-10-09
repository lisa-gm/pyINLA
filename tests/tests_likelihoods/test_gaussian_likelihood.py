import numpy as np
from autograd import grad, hessian
from scipy.stats import multivariate_normal

from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.core.pyinla_config import PyinlaConfig

pyinla_config = PyinlaConfig()


def test_gaussian_evaluate_likelihood(
    generate_gaussian_data,
    n_observations: int,
    theta_observations: float,
):
    a, x, y, theta_likelihood = generate_gaussian_data

    likelihood_instance = GaussianLikelihood(pyinla_config, n_observations)

    eta = a @ x
    likelihood_inla = likelihood_instance.evaluate_likelihood(eta, y, theta_likelihood)

    # Reference using scipy.stats.multivariate_normal
    Sigma = 1 / np.exp(theta_observations) * np.eye(n_observations)
    multivariate_normal_ref = multivariate_normal.logpdf(y, mean=eta, cov=Sigma)
    # Deduct constant
    multivariate_normal_ref += 0.5 * n_observations * np.log(2 * np.pi)

    assert np.allclose(likelihood_inla, multivariate_normal_ref)


def test_gaussian_evaluate_gradient(
    generate_gaussian_data,
    n_observations: int,
    n_latent_parameters: int,
    theta_observations: float,
):
    a, x, y, theta_likelihood = generate_gaussian_data
    eta = a @ x

    likelihood_instance = GaussianLikelihood(pyinla_config, n_observations)

    grad_likelihood_inla = likelihood_instance.evaluate_gradient_likelihood(
        eta, y, theta_likelihood
    )

    auto_grad = grad(likelihood_instance.evaluate_likelihood, 0)
    grad_likelihood_ref = auto_grad(eta, y, theta_likelihood)

    # assert np.allclose(2, 2)
    assert np.allclose(grad_likelihood_inla, grad_likelihood_ref)


def test_gaussian_evaluate_hessian(
    generate_gaussian_data,
    n_observations: int,
    n_latent_parameters: int,
    theta_observations: float,
):
    a, x, y, theta_likelihood = generate_gaussian_data
    eta = a @ x

    likelihood_instance = GaussianLikelihood(pyinla_config, n_observations)

    hessian_likelihood_inla = likelihood_instance.evaluate_hessian_likelihood(
        eta, y, theta_likelihood
    )
    hessian_likelihood_inla = hessian_likelihood_inla.toarray()

    auto_hessian = hessian(likelihood_instance.evaluate_likelihood, 0)
    hessian_likelihood_ref = auto_hessian(eta, y, theta_likelihood)

    assert np.allclose(hessian_likelihood_inla, hessian_likelihood_ref)
