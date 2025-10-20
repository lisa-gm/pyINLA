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
):
    from dalia.configs.dalia_config import SolverConfig
    from dalia.solvers import SparseSolver

    config = SolverConfig(type=solver_type)

    if solver_type == "scipy":
        return SparseSolver(config=config)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


def _generate_spd_spmatrix(
    n: int,
    density: float,
):
    """Returns a random, positive definite, sparse matrix.

    Parameters
    ----------
    n : int
        Size of the matrix.
    density : float
        Density of the matrix.

    Returns
    -------
    A : ArrayLike
        Random, positive definite, sparse matrix.
    """
    L = sparse.random(n, n, density=density, data_rvs=np.random.randn)
    L = L + n * sparse.eye(n)  # Make diagonal dominant
    L = sparse.tril(L)  # lower triangular

    return L @ L.T  # SPD and sparse (but denser than L)
