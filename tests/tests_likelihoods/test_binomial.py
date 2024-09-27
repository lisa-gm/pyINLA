import numpy as np
from scipy import sparse
from scipy.special import gammaln
from scipy.stats import binom

from pyinla.likelihoods.binomial import BinomialLikelihood
from pyinla.utils import sigmoid


def test_binomial_evaluate_likelihood(
    n_observations: int,
    n_latent_parameters: int,
    theta_observations: float,
    pyinla_config,
):
    theta_likelihood: dict = {"theta_observations": theta_observations}

    a = sparse.random(n_observations, n_latent_parameters, density=0.5)
    x = np.random.randn(n_latent_parameters)
    eta = a @ x

    n_trials = np.ones(len(eta), dtype=int)
    prob = sigmoid(eta)
    y = np.random.binomial(n=n_trials, p=prob)

    likelihood_instance = BinomialLikelihood(pyinla_config, n_observations)

    # Neglects constant in likelihood (w/o  - sum(log(y!)))
    likelihood_inla = likelihood_instance.evaluate_likelihood(y, eta, theta_likelihood)
    print(f"likelihood_inla: {likelihood_inla}")

    # Reference using scipy.stats.binom -> correct for the constant
    binom_ref_vec = binom.logpmf(y, p=prob, n=n_trials)
    binom_const = gammaln(n_trials + 1) - (gammaln(y + 1) + gammaln(n_trials - y + 1))
    binom_ref = binom_ref_vec.sum() - binom_const.sum()
    print(f"poisson_ref: {binom_ref}")

    assert np.allclose(likelihood_inla, binom_ref)


def test_binomial_evaluate_gradient(): ...


def test_binomial_evaluate_hessian(): ...
