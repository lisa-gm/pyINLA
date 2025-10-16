# Copyright 2024-2025 DALIA authors. All rights reserved.

import numpy as np
import pytest
from dalia import backend_flags, xp, ArrayLike

SEED = 63

np.random.seed(SEED)

if backend_flags["cupy_avail"]:
    import cupy as cp

    cp.random.seed(cp.uint64(SEED))


def _create_solver(
    solver_type: str,
    diagonal_blocksize: int,
    n_diag_blocks: int,
    arrowhead_blocksize: int = 0,
):
    from dalia.configs.dalia_config import SolverConfig
    from dalia.solvers import SerinvSolver

    config = SolverConfig(type=solver_type)

    if solver_type == "serinv":
        return SerinvSolver(
            config=config,
            diagonal_blocksize=diagonal_blocksize,
            n_diag_blocks=n_diag_blocks,
            arrowhead_blocksize=arrowhead_blocksize,
        )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
    
    
def _create_pobta(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
):
    """Returns a random, positive definite, block tridiagonal arrowhead matrix.

    Parameters
    ----------
    diagonal_blocksize : int
        Size of the diagonal blocks.
    arrowhead_blocksize : int
        Size of the arrowhead blocks.
    n_diag_blocks : int
        Number of diagonal blocks.

    Returns
    -------
    A : ArrayLike
        Random, positive definite, block tridiagonal arrowhead matrix.
    """

    A = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=xp.float64,
    )

    # Fill the arrowhead blocks
    A[-arrowhead_blocksize:, :-arrowhead_blocksize] = xp.random.rand(
        arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
    )
    A[-arrowhead_blocksize:, -arrowhead_blocksize:] = xp.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        # Fill the off-diagonal blocks
        if i < n_diag_blocks - 1:
            A[
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ] = xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make symmetric
    A = A + A.T
    
    # Make positive definite by adding scaled identity
    A = A + (xp.max(xp.abs(A)) + 1.0) * xp.eye(A.shape[0])

    return A


def _create_pobt(
    diagonal_blocksize: int,
    n_diag_blocks: int,
):
    """Returns a random, positive definite, block tridiagonal matrix.

    Parameters
    ----------
    diagonal_blocksize : int
        Size of the diagonal blocks.
    n_diag_blocks : int
        Number of diagonal blocks.

    Returns
    -------
    A : ArrayLike
        Random, positive definite, block tridiagonal matrix.
    """

    A = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks,
            diagonal_blocksize * n_diag_blocks,
        ),
        dtype=xp.float64,
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        # Fill the off-diagonal blocks
        if i < n_diag_blocks - 1:
            A[
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ] = xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make symmetric
    A = A + A.T
    
    # Make positive definite by adding scaled identity
    A = A + (xp.max(xp.abs(A)) + 1.0) * xp.eye(A.shape[0])

    return A


def _allclose_dense_structured(
    A: ArrayLike, 
    B: ArrayLike, 
    diagonal_blocksize: int, 
    n_diag_blocks: int, 
    arrowhead_blocksize: int = 0):
    """Check block-wise correctness of two structured matrices in dense storage format.

    Parameters
    ----------
    A : ArrayLike
        First structured matrix to compare.
    B : ArrayLike
        Second structured matrix to compare.
    sparsity : str
        Sparsity pattern, either "bt" or "bta".
    diagonal_blocksize : int
        Size of the diagonal blocks.
    n_diag_blocks : int
        Number of diagonal blocks.
    arrowhead_blocksize : int, optional
        Size of the arrowhead blocks, by default 0.
    """ 
    if arrowhead_blocksize > 0:
        np.allclose(
            A[-arrowhead_blocksize:, :-arrowhead_blocksize],
            B[-arrowhead_blocksize:, :-arrowhead_blocksize],
            rtol=1e-14,
            atol=1e-16,
        )

        np.allclose(
            A[-arrowhead_blocksize:, -arrowhead_blocksize:],
            B[-arrowhead_blocksize:, -arrowhead_blocksize:],
            rtol=1e-14,
            atol=1e-16,
        )

    # Check the diagonal blocks
    for i in range(n_diag_blocks):
        np.allclose(
            A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ],
            B[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ],
            rtol=1e-14,
            atol=1e-16,
        )

        # Check the off-diagonal blocks
        if i < n_diag_blocks - 1:
            np.allclose(
                A[
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                ],
                B[
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                ],
                rtol=1e-14,
                atol=1e-16,
            )
    