# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
import pytest
from scipy import sparse

from pyinla.core.solver import Solver


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [0, 1, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3])
def test_get_selected_inverse(
    solver: Solver,
    pobta_dense,
    pyinla_config,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):
    A_coo = sparse.coo_matrix(pobta_dense)

    A_inv_ref = np.linalg.inv(pobta_dense)
    A_inv_ref_selected = sparse.lil_matrix(A_coo.shape)
    for i in range(len(A_coo.row)):
        A_inv_ref_selected[A_coo.row[i], A_coo.col[i]] = A_inv_ref[
            A_coo.row[i], A_coo.col[i]
        ]

    solver_instance = solver(
        pyinla_config, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )
    solver_instance.cholesky(A_coo)
    solver_instance.full_inverse()

    A_inv_sparse_solver = solver_instance.get_selected_inverse()

    assert np.allclose(A_inv_ref_selected.toarray(), A_inv_sparse_solver.toarray())
