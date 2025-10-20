# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
from scipy import sparse

from dalia import backend_flags, xp

SEED = 63

np.random.seed(SEED)

if backend_flags["cupy_avail"]:
    import cupy as cp

    cp.random.seed(cp.uint64(63))


def _create_solver(
    solver_type: str,
    matrix_size: int,
):
    from dalia.configs.dalia_config import SolverConfig
    from dalia.solvers import DenseSolver

    config = SolverConfig(type=solver_type)

    if solver_type == "dense":
        return DenseSolver(config=config, n=matrix_size)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


def _generate_spd_spmatrix(
    matrix_type: str,
    n: int,
):
    """Returns a random, positive definite, matrix.

    Parameters
    ----------
    matrix_type : str
        Type of the matrix: "sparse" or "dense".
    n : int
        Size of the matrix.

    Returns
    -------
    A : ArrayLike
        Random, positive definite, matrix.
    """

    if matrix_type == "sparse":
        L = sparse.random(n, n, density=0.5, data_rvs=np.random.randn)
        L = L + n * sparse.eye(n)  # Make diagonal dominant
        L = sparse.tril(L)  # lower triangular

        return L @ L.T  # SPD and sparse (but denser than L)
    else:
        L = np.tril(np.random.rand(n, n))
        L += np.diag(np.sum(np.abs(L), axis=1))
        return L @ L.T  # SPD and dense
