# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
import pytest
from scipy import sparse

from pyinla.core.solver import Solver


@pytest.mark.parametrize("diagonal_blocksize", [0, 2, 3])  #
@pytest.mark.parametrize("arrowhead_blocksize", [1, 3])
@pytest.mark.parametrize("n_diag_blocks", [0, 1, 3])  #
def test_solve(
    solver: Solver,
    pobta_dense,
    pyinla_config,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):
    rhs = np.random.rand(pobta_dense.shape[0])

    X_ref = np.linalg.solve(pobta_dense, rhs)

    A_csr = sparse.csr_matrix(pobta_dense)
    solver_instance = solver(
        pyinla_config, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    sparsity = "bta"
    if diagonal_blocksize == 0 or n_diag_blocks == 0:
        sparsity = "d"

    solver_instance.cholesky(A_csr, sparsity=sparsity)
    X_solver = solver_instance.solve(rhs, sparsity=sparsity)

    assert np.allclose(X_solver, X_ref)
