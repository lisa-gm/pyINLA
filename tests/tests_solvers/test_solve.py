# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
import pytest
from scipy import sparse

from pyinla.core.solver import Solver


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [0, 1, 2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3, 4])
def test_solve(
    solver: Solver,
    pobta_dense,
    pyinla_config,
):
    rhs = np.random.rand(pobta_dense.shape[0])

    X_ref = np.linalg.solve(pobta_dense, rhs)

    A_csr = sparse.csr_matrix(pobta_dense)
    solver_instance = solver(pyinla_config)
    solver_instance.cholesky(A_csr)

    X_solver = solver_instance.solve(rhs)

    assert np.allclose(X_solver, X_ref)
