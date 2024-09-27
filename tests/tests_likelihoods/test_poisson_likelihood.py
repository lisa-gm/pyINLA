import numpy as np
import autograd.numpy as anp
from autograd import grad, hessian
from scipy.special import gammaln
from scipy.stats import poisson

from pyinla.likelihoods.poisson import PoissonLikelihood


def test_poisson_evaluate_likelihood(
    generate_poisson_data,
    n_observations: int,
    pyinla_config,
):

    y, eta, lam, theta_likelihood = generate_poisson_data
    likelihood_instance = PoissonLikelihood(pyinla_config, n_observations)

    # Neglects constant in likelihood (w/o  - sum(log(y!)))
    likelihood_inla = likelihood_instance.evaluate_likelihood(y, eta, theta_likelihood)

    # Reference using scipy.stats.poisson -> correct for the constant
    poisson_ref_vec = poisson.logpmf(y, mu=lam)
    poisson_ref = poisson_ref_vec.sum() + gammaln(y + 1).sum()

    assert np.allclose(likelihood_inla, poisson_ref)


def test_poisson_evaluate_gradient(
    generate_poisson_data,
    n_observations: int,
    pyinla_config,
):

    y, eta, lam, theta_likelihood = generate_poisson_data
    likelihood_instance = PoissonLikelihood(pyinla_config, n_observations)

    grad_likelihood_inla = likelihood_instance.evaluate_gradient_likelihood(
        y, eta, theta_likelihood
    )

    eta_anp = anp.array(eta)
    auto_grad = grad(likelihood_instance.evaluate_likelihood_autodiff, 1)
    grad_likelihood_ref = auto_grad(y, eta_anp, theta_likelihood)

    assert np.allclose(grad_likelihood_inla, grad_likelihood_ref)


def test_poisson_evaluate_hessian(
    generate_poisson_data,
    n_observations: int,
    pyinla_config,
):

    y, eta, _, theta_likelihood = generate_poisson_data
    likelihood_instance = PoissonLikelihood(pyinla_config, n_observations)

    hessian_likelihood_inla = likelihood_instance.evaluate_hessian_likelihood(
        y, eta, theta_likelihood
    )
    hessian_likelihood_inla = hessian_likelihood_inla.toarray()

    eta_anp = anp.array(eta)
    auto_hessian = hessian(likelihood_instance.evaluate_likelihood_autodiff, 1)
    hessian_likelihood_ref = auto_hessian(y, eta_anp, theta_likelihood)

    assert np.allclose(hessian_likelihood_inla, hessian_likelihood_ref)
