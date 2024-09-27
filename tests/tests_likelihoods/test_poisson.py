import numpy as np
from scipy import sparse
from scipy.special import gammaln
from scipy.stats import poisson

from pyinla.likelihoods.poisson import PoissonLikelihood


def test_poisson(
    n_observations: int,
    n_latent_parameters: int,
    theta_observations: float,
    pyinla_config,
):
    theta_likelihood: dict = {"theta_observations": theta_observations}

    a = sparse.random(n_observations, n_latent_parameters, density=0.5)
    x = np.random.randn(n_latent_parameters)
    eta = a @ x
    lam = np.exp(eta)
    y = np.random.poisson(lam=lam)

    likelihood_instance = PoissonLikelihood(pyinla_config, n_observations)

    # Neglects constant in likelihood (w/o  - sum(log(y!)))
    likelihood_inla = likelihood_instance.evaluate_likelihood(y, eta, theta_likelihood)

    # Reference using scipy.stats.poisson -> correct for the constant
    poisson_ref_vec = poisson.logpmf(y, mu=lam)
    poisson_ref = poisson_ref_vec.sum() + gammaln(y + 1).sum()

    assert np.allclose(likelihood_inla, poisson_ref)
