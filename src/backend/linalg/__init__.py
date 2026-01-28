# src/backend/linalg/__init__.py - Public API for users

from backend.linalg.solvers import (
    DenseSolver,
    LinearSolver,
    SparseSolver,
)


def create_solver(matrix, overwrite_matrix=False) -> LinearSolver:
    """Factory for creating appropriate linear solver."""
    # pylint: disable=import-outside-toplevel
    from backend.datastructures import DenseMatrix, SparseMatrix

    if isinstance(matrix, SparseMatrix):
        return SparseSolver(matrix, overwrite_matrix=overwrite_matrix)
    if isinstance(matrix, DenseMatrix):
        return DenseSolver(matrix, overwrite_matrix=overwrite_matrix)
    raise TypeError(f"Unknown matrix type: {type(matrix)}")


__all__ = ["create_solver", "LinearSolver", "DenseSolver", "SparseSolver"]
