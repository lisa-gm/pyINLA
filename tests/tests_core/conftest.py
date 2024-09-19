# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest

from pyinla.core.pyinla_config import PyinlaConfig

SEED = 63

np.random.seed(SEED)


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
def pyinla_config(inla):
    """Returns a PyinlaConfig object."""

    pyinla_config = PyinlaConfig()
    pyinla_config.solver.type = "scipy"

    return pyinla_config
