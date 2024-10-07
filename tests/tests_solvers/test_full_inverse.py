# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
import pytest
from scipy import sparse

from pyinla.core.solver import Solver


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [0, 1, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3])
def test_cholesky(
    solver: Solver,
    pobta_dense,
    pyinla_config,
):
    A_csc = sparse.csr_matrix(pobta_dense)

    A_inv_ref = np.linalg.inv(pobta_dense)

    solver_instance = solver(pyinla_config)
    solver_instance.cholesky(A_csc)
    solver_instance.full_inverse()

    A_inv_solver = solver_instance.A_inv

    assert np.allclose(A_inv_solver, A_inv_ref)
