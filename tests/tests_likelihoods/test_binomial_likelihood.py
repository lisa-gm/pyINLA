import autograd.numpy as anp
import numpy as np
from autograd import grad, hessian
from scipy.special import gammaln
from scipy.stats import binom

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.likelihoods.binomial import BinomialLikelihood

pyinla_config = PyinlaConfig()


def test_binomial_evaluate_likelihood(
    generate_binomial_data,
    n_observations: int,
    n_latent_parameters: int,
    theta_observations: float,
):
    eta, y, n_trials, prob, theta_likelihood = generate_binomial_data

    likelihood_instance = BinomialLikelihood(pyinla_config, n_observations)
    likelihood_instance.n_trials = n_trials

    # Neglects constant in likelihood (w/o  - sum(log(y!)))
    likelihood_inla = likelihood_instance.evaluate_likelihood(eta, y, theta_likelihood)
    print(f"likelihood_inla: {likelihood_inla}")

    # Reference using scipy.stats.binom -> correct for the constant
    binom_ref_vec = binom.logpmf(y, p=prob, n=n_trials)
    binom_const = gammaln(n_trials + 1) - (gammaln(y + 1) + gammaln(n_trials - y + 1))
    binom_ref = binom_ref_vec.sum() - binom_const.sum()
    print(f"poisson_ref: {binom_ref}")

    assert np.allclose(likelihood_inla, binom_ref)


def test_binomial_evaluate_gradient(
    generate_binomial_data,
    n_observations: int,
):
    eta, y, n_trials, prob, theta_likelihood = generate_binomial_data
    likelihood_instance = BinomialLikelihood(pyinla_config, n_observations)
    likelihood_instance.n_trials = n_trials

    grad_likelihood_inla = likelihood_instance.evaluate_gradient_likelihood(
        eta, y, theta_likelihood
    )

    eta_anp = anp.array(eta)
    auto_grad = grad(likelihood_instance.evaluate_likelihood_autodiff, 0)
    grad_likelihood_ref = auto_grad(eta_anp, y, theta_likelihood)

    assert np.allclose(grad_likelihood_inla, grad_likelihood_ref)


def test_binomial_evaluate_hessian(
    generate_binomial_data,
    n_observations: int,
):
    eta, y, n_trials, prob, theta_likelihood = generate_binomial_data
    likelihood_instance = BinomialLikelihood(pyinla_config, n_observations)
    likelihood_instance.n_trials = n_trials

    hessian_likelihood_inla = likelihood_instance.evaluate_hessian_likelihood(
        eta, y, theta_likelihood
    )

    hessian_likelihood_inla = hessian_likelihood_inla.toarray()

    eta_anp = anp.array(eta)
    auto_hessian = hessian(likelihood_instance.evaluate_likelihood_autodiff, 0)
    hessian_likelihood_ref = auto_hessian(eta_anp, y, theta_likelihood)

    assert np.allclose(hessian_likelihood_inla, hessian_likelihood_ref)
