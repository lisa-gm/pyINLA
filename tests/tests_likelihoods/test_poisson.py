import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import eye
import math


# from scipy.stats import multivariate_normal
from scipy.stats import poisson
from scipy.special import gammaln

from pyinla.core.likelihood import Likelihood


@pytest.mark.parametrize("n_observations", [1, 2, 3, 9])
@pytest.mark.parametrize("n_latent_parameters", [1, 2, 3, 8])
@pytest.mark.parametrize("theta_observations", [-0.1, 0.0, 0.1, 0.2])
def test_poisson(
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
    lam = np.exp(eta)
    y = np.random.poisson(lam=lam)

    likelihood_instance = likelihood(pyinla_config, n_observations)

    # neglects constant in likelihood (w/o  - sum(log(y!)))
    likelihood_inla = likelihood_instance.evaluate_likelihood(y, eta, theta_likelihood)
    # print(f"likelihood_inla: {likelihood_inla}")

    # reference using scipy.stats.poisson -> correct for the constant
    poisson_ref_vec = poisson.logpmf(y, mu=lam)
    poisson_ref = poisson_ref_vec.sum() + gammaln(y + 1).sum()
    # print(f"poisson_ref: {poisson_ref}")

    assert np.allclose(likelihood_inla, poisson_ref)
