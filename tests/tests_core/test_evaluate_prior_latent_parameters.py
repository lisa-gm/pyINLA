# Copyright 2024 pyINLA authors. All rights reserved.

import pytest
import numpy as np
from scipy import sparse

from scipy.stats import multivariate_normal

from pyinla.core import INLA


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [0, 1, 2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3, 4])
def test_evaluate_prior_latent_parameters(
    inla: INLA,
    pobta_dense,
    pyinla_config,
):

    inla_instance = inla(pyinla_config)

    Q_prior = sparse.csr_matrix(pobta_dense)
    x_star = np.random.randn(pobta_dense.shape[0])

    # invert Q_prior
    Sigma_prior = np.linalg.inv(pobta_dense)
    log_prior_ref = multivariate_normal.logpdf(
        x_star, mean=np.zeros(pobta_dense.shape[0]), cov=Sigma_prior
    )
    # evaluate multivariate normal density at x_star given Q_prior
    log_prior_inla = inla._evaluate_prior_latent_parameters(Q_prior, x_star)

    assert np.allclose(log_prior_inla, log_prior_ref)
