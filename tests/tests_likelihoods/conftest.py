# write functions that generate small test cases

import pytest

from pyinla.core.pyinla_config import PyinlaConfig

# from pyinla.likelihoods.binomial import BinomialLikelihood
# from pyinla.likelihoods.gaussian import GaussianLikelihood
# from pyinla.likelihoods.poisson import PoissonLikelihood

# LIKELIHOODS = [BinomialLikelihood, GaussianLikelihood, PoissonLikelihood]  #


# @pytest.fixture(params=LIKELIHOODS, autouse=True)
# def likelihood(request):
#     return request.param


@pytest.fixture
def theta_likelihood():
    return {"theta_observations": -0.1}


@pytest.fixture(scope="function", autouse=False)
def pyinla_config(likelihood):
    """Returns a PyinlaConfig object."""

    pyinla_config = PyinlaConfig()

    return pyinla_config
