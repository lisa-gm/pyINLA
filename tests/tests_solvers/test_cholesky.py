# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
import pytest
from scipy import sparse

from pyinla.core.solver import Solver


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [1, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3])
def test_cholesky(
    solver: Solver,
    pobta_dense,
    pyinla_config,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):

    L_ref = np.linalg.cholesky(pobta_dense)

    A_csr = sparse.csr_matrix(pobta_dense)
    solver_instance = solver(
        pyinla_config, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )
    solver_instance.cholesky(A_csr, sparsity="bta")
    L_solver = solver_instance.get_L(sparsity="bta")

    assert np.allclose(L_solver, L_ref)
