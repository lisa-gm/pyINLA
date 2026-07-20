# backend/linalg/solvers/__init__.py - Internal API for solvers module
from .dense.dense_solver import DenseSolver
from .linear_solver import LinearSolver
from .sparse.cudss import CuDSS
from .sparse.sparse_solver import SparseSolver


def linear_solver_factory(matrix, overwrite_matrix=False) -> LinearSolver:
    """Factory for creating appropriate linear solver."""
    # pylint: disable=import-outside-toplevel
    from dalia.backend.datastructures import DenseMatrix, SparseMatrix

    if isinstance(matrix, SparseMatrix):
        return SparseSolver(matrix, overwrite_matrix=overwrite_matrix)
    if isinstance(matrix, DenseMatrix):
        return DenseSolver(matrix, overwrite_matrix=overwrite_matrix)
    raise TypeError(f"Unknown matrix type: {type(matrix)}")


__all__ = [
    "linear_solver_factory",
    "LinearSolver",
    "DenseSolver",
    "SparseSolver",
    "CuDSS",
]
