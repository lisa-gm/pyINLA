# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import backend_flags, xp

import numpy as np

SEED = 63

np.random.seed(SEED)

if backend_flags["cupy_avail"]:
    import cupy as cp

    cp.random.seed(cp.uint64(63))

def _rhs(
    n_rhs: int,
    matrix_size: int
):
    """Returns a random right-hand side.

    Parameters
    ----------
    n_rhs : int
        Number of right-hand sides.
    diagonal_blocksize : int
        Size of the diagonal blocks.
    arrowhead_blocksize : int
        Size of the arrowhead blocks.
    n_diag_blocks : int
        Number of diagonal blocks.

    Returns
    -------
    B : ArrayLike
        Random right-hand side.
    """

    B = xp.random.rand(
        matrix_size, n_rhs
    )

    return B


def _reference_cholesky(A):
    """Compute reference Cholesky decomposition using NumPy.
    
    Parameters
    ----------
    A : ArrayLike
        Input matrix to decompose.
        
    Returns
    -------
    L : numpy.ndarray
        Lower triangular Cholesky factor.
    """
    A_dense = A.toarray() if hasattr(A, 'toarray') else A
    A_dense = np.asarray(A_dense)
    return np.linalg.cholesky(A_dense)


def _reference_solve(A, rhs):
    """Solve linear system using NumPy.
    
    Parameters
    ----------
    A : ArrayLike
        System matrix.
    rhs : ArrayLike
        Right-hand side.
        
    Returns
    -------
    x : numpy.ndarray
        Solution vector.
    """
    A_dense = A.toarray() if hasattr(A, 'toarray') else A
    A_dense = np.asarray(A_dense)
    rhs_dense = np.asarray(rhs)
    return np.linalg.solve(A_dense, rhs_dense)


def _reference_logdet(A):
    """Compute log determinant using NumPy determinant.
    
    Parameters
    ----------
    A : ArrayLike
        Input matrix.
        
    Returns
    -------
    logdet : float
        Log determinant of the matrix.
    """
    A_dense = A.toarray() if hasattr(A, 'toarray') else A
    A_dense = np.asarray(A_dense)
    # Use numpy's determinant for the most reliable reference
    return np.log(np.linalg.det(A_dense))


def _reference_inversion(A):
    """Compute matrix inverse using NumPy.
    
    Parameters
    ----------
    A : ArrayLike
        Input matrix.
        
    Returns
    -------
    A_inv : numpy.ndarray
        Inverse matrix.
    """
    A_dense = A.toarray() if hasattr(A, 'toarray') else A
    A_dense = np.asarray(A_dense)
    return np.linalg.inv(A_dense)
