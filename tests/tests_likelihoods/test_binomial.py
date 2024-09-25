import numpy as np
import pytest
from scipy import sparse
from scipy.special import gammaln

# from scipy.stats import multivariate_normal
from scipy.stats import binom

from pyinla.core.likelihood import Likelihood
from pyinla.likelihoods.binomial import BinomialLikelihood
from pyinla.utils import sigmoid


@pytest.mark.parametrize("likelihood", [BinomialLikelihood])  #
@pytest.mark.parametrize("n_observations", [1, 2, 3, 9])  #
@pytest.mark.parametrize("n_latent_parameters", [1, 2, 3, 8])  #
@pytest.mark.parametrize("theta_observations", [-0.1, 0.0, 0.1, 0.2])  #
def test_binomial(
    likelihood: Likelihood,
    n_observations: int,
    n_latent_parameters: int,
    theta_likelihood: dict,
    theta_observations: float,
    pyinla_config,
):
    theta_likelihood["theta_observations"] = theta_observations

    a = sparse.random(n_observations, n_latent_parameters, density=0.5)
    x = np.random.randn(n_latent_parameters)
    eta = a @ x

    n_trials = np.ones(len(eta), dtype=int)
    # prob = np.exp(eta) / (1 + np.exp(eta))
    prob = sigmoid(eta)
    y = np.random.binomial(n=n_trials, p=prob)

    likelihood_instance = likelihood(pyinla_config, n_observations)

    # neglects constant in likelihood (w/o  - sum(log(y!)))
    likelihood_inla = likelihood_instance.evaluate_likelihood(y, eta, theta_likelihood)
    print(f"likelihood_inla: {likelihood_inla}")

    # reference using scipy.stats.binom -> correct for the constant
    binom_ref_vec = binom.logpmf(y, p=prob, n=n_trials)
    binom_const = gammaln(n_trials + 1) - (gammaln(y + 1) + gammaln(n_trials - y + 1))
    binom_ref = binom_ref_vec.sum() - binom_const.sum()
    print(f"poisson_ref: {binom_ref}")

    assert np.allclose(likelihood_inla, binom_ref)
