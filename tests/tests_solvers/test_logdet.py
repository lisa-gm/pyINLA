# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
import pytest
from numpy.linalg import slogdet
from scipy import sparse

from pyinla.core.solver import Solver


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [0, 1, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3])
def test_logdet(
    solver: Solver,
    pobta_dense,
    pyinla_config,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):
    sign, absLogdet = slogdet(pobta_dense)

    A_csr = sparse.csr_matrix(pobta_dense)
    solver_instance = solver(
        pyinla_config, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )
    solver_instance.cholesky(A_csr)
    logdet_solver = solver_instance.logdet()

    assert np.allclose(logdet_solver, sign * absLogdet)
