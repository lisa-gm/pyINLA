# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest

from numpy.typing import ArrayLike

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.solvers.scipy_solver import ScipySolver

from pyinla.core.likelihood import Likelihood
from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.likelihoods.poisson import PoissonLikelihood
from pyinla.likelihoods.binomial import BinomialLikelihood

from pyinla.core.model import Model
from pyinla.models.regression import RegressionModel
from pyinla.models.spatio_temporal import SpatioTemporalModel

SOLVER = [ScipySolver]


@pytest.fixture(params=SOLVER, autouse=True)
def solver(request):
    return request.param


@pytest.fixture(scope="function", autouse=False)
def pobta_dense(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
):
    """Returns a random, positive definite, block tridiagonal arrowhead matrix."""

    pobta_dense = np.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=np.float64,
    )

    if arrowhead_blocksize != 0:
        # Fill the lower arrowhead blocks
        pobta_dense[-arrowhead_blocksize:, :-arrowhead_blocksize] = np.random.rand(
            arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
        )

        # Fill the tip of the arrowhead
        pobta_dense[-arrowhead_blocksize:, -arrowhead_blocksize:] = np.random.rand(
            arrowhead_blocksize, arrowhead_blocksize
        )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        pobta_dense[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = np.random.rand(diagonal_blocksize, diagonal_blocksize) + np.eye(
            diagonal_blocksize
        )

        # Fill the off-diagonal blocks
        if i > 0:
            pobta_dense[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ] = np.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make the matrix diagonally dominant
    for i in range(pobta_dense.shape[0]):
        pobta_dense[i, i] = 1 + np.sum(pobta_dense[i, :])

    # Make the matrix symmetric
    pobta_dense = (pobta_dense + pobta_dense.T) / 2

    return pobta_dense


@pytest.fixture(scope="function", autouse=False)
def pyinla_config(solver):
    """Returns a PyinlaConfig object."""

    pyinla_config = PyinlaConfig()

    if solver == "ScipySolver":
        pyinla_config.solver.type = "scipy"
    elif solver == "SerinvSolver":
        pyinla_config.solver.type = "serinv"

    return pyinla_config


@pytest.fixture(scope="function", autouse=False)
def pyinla_config_initialize_theta(
    model: Model,
    likelihood: Likelihood,
    theta_prior_mean: ArrayLike,
    theta_prior_variance: ArrayLike,
):

    pyinla_config = PyinlaConfig()

    counter = 0
    if Model == RegressionModel and Likelihood == GaussianLikelihood:
        pyinla_config.prior_hyperparamters.mean_theta_observations = theta_prior_mean[
            counter
        ]
        pyinla_config.prior_hyperparamters.precision_theta_observations = (
            theta_prior_variance[counter]
        )
        counter += 1

    elif Model == SpatioTemporalModel:
        pyinla_config.prior_hyperparamters.mean_theta_spatial_range = theta_prior_mean[
            counter
        ]
        pyinla_config.prior_hyperparamters.precision_theta_spatial_range = (
            theta_prior_variance[counter]
        )
        counter += 1

        pyinla_config.prior_hyperparamters.mean_theta_temporal_range = theta_prior_mean[
            counter
        ]
        pyinla_config.prior_hyperparamters.precision_theta_temporal_range = (
            theta_prior_variance[counter]
        )
        counter += 1

        pyinla_config.prior_hyperparamters.mean_theta_spatio_temporal_variation = (
            theta_prior_mean[counter]
        )
        pyinla_config.prior_hyperparamters.precision_theta_spatio_temporal_variation = (
            theta_prior_variance[counter]
        )
        counter += 1

    else:
        print("undefined model!")
        raise ValueError

    return pyinla_config
