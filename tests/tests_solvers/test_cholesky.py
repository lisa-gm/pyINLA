# Copyright 2024 pyINLA authors. All rights reserved.

import pytest
import numpy as np
from scipy import sparse


from pyinla.core.solver import Solver
from pyinla.solvers.scipy_solver import ScipySolver


@pytest.mark.parametrize("solver", [ScipySolver])
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [0, 1, 2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3, 4])
def test_cholesky(
    solver: Solver,
    pobta_dense,
    pyinla_config,
):
    L_ref = np.linalg.cholesky(pobta_dense)

    A_csr = sparse.csr_matrix(pobta_dense)
    solver_instance = solver(pyinla_config)
    solver_instance.cholesky(A_csr)
    L_solver = solver_instance.L.toarray()

    assert np.allclose(L_solver, L_ref)
