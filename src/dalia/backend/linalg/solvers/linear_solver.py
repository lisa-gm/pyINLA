# src/dalia/backend/linalg/solvers/linear_solver.py
"""
Base class for linear solvers.

Each linear solvers should provide the following functionalities:
- Return the factorization, either Cholesky, LDL^T, or LU, based on the matrix properties.
- Solve a linear system Ax = b using the chosen factorization.
- Perform the selected-inversion of a matrix A given it's factors (matching the sparsity pattern of A).

"""

from abc import ABC, abstractmethod

from dalia.backend.datastructures import Matrix


class LinearSolver(ABC):
    """Abstract base class for linear solvers.

    Provides template methods for common operations with state management.
    Subclasses implement wraps library calls for specific matrix types.
    """

    # 1. Class attributes (if any)
    # 2. Initialization

    def __init__(self, matrix: Matrix, overwrite_matrix: bool = False):
        """
        Parameters
        ----------
        matrix : Matrix
            System matrix to solve with
        overwrite_matrix : bool
            If True, allows in-place factorization that destroys original matrix data.
            Use when matrix is no longer needed after factorization to save memory.
        """
        self._matrix: Matrix = matrix
        self._overwrite_matrix: bool = overwrite_matrix
        self._is_factorized: bool = False
        self._factors: Matrix = None
        self._target = self._choose_target()

    # 3. Special representation methods
    # 4. Properties (grouped together)
    # 5. Comparison operators (if needed)
    # 6. Arithmetic operators (standard order)
    # 7. Right-hand operators (same order as above)
    # 8. In-place operators (if supported)
    # 9. Other special methods
    # 10. Public methods

    def update_matrix(self, new_matrix: Matrix):
        """Update system matrix and invalidate factorization.

        Parameters
        ----------
        new_matrix : Matrix
            New system matrix. Must be same type and shape.
            For sparse matrices, should ideally have same sparsity pattern.
        """
        if not isinstance(new_matrix, type(self._matrix)):
            raise TypeError("Cannot change matrix type after solver creation")
        if new_matrix.shape != self._matrix.shape:
            raise ValueError("Cannot change matrix shape after solver creation")

        self._matrix: Matrix = new_matrix
        self._is_factorized: bool = False  # Invalidate factorization
        self._target = self._choose_target()  # Update target if needed

    def factorize(self, matrix: Matrix = None, overwrite: bool = None):
        """Factorize system matrix.

        Parameters
        ----------
        matrix : Matrix, optional
            If provided, updates system matrix before factorizing.
        overwrite : bool, optional
            Override overwrite_matrix setting for this factorization.
            If True, original matrix data may be destroyed.

        Warnings
        --------
        If overwrite=True, the input matrix's ._data will be overwritten
        with factorization data. The Matrix object should not be used
        afterward except through the solver. In particular:
        - matrix._data will contain factor(s), not original matrix
        - matrix.toarray() will return factors, not original matrix

        Examples
        --------
        Safe usage (recommended):
        >>> Q = build_precision_matrix()
        >>> solver.factorize(Q, overwrite=True)
        >>> # Don't use Q anymore - it's corrupted

        >>> Q_new = build_precision_matrix()  # Build fresh
        >>> solver.factorize(Q_new, overwrite=True)
        """
        if matrix is not None:
            self.update_matrix(matrix)

        if overwrite is None:
            overwrite: bool = self._overwrite_matrix

        self._factors = self._compute_factorization(overwrite=overwrite)
        self._is_factorized: bool = True

    def solve(self, b):
        """Solve linear system Ax = b.

        Parameters
        ----------
        b : array-like
            Right-hand side vector or matrix.

        Returns
        -------
        x : array-like
            Solution vector or matrix.
        """
        if not self._is_factorized:
            raise RuntimeError("Matrix must be factorized before solving")

        if isinstance(b, Matrix):
            b_data = b.toarray()
        else:
            b_data = b

        return self._solve_system(b_data)

    def logdet(self) -> float:
        """Compute log-determinant of the matrix.

        Returns
        -------
        float
            Log-determinant of the matrix.
        """
        if not self._is_factorized:
            raise RuntimeError("Matrix must be factorized before computing logdet")
        return self._compute_logdet()

    def selected_inverse(self, overwrite_factors: bool = False) -> Matrix:
        """Compute selected inverse matching sparsity pattern.

        Parameters
        ----------
        overwrite_factors : bool, optional
            If True, destroys stored factorization to save memory.
            The solver becomes UNUSABLE after this call (cannot solve() again)
            before updating matrix and re-factorizing.
            Use when inverse is the final operation with the current factors.
            Default: False (keeps factorization and return a new Matrix).

        Returns
        -------
        Matrix
            Inverse matrix (type matches solver: DenseMatrix or SparseMatrix).

        Warnings
        --------
        When overwrite_factors=True:
        - Factorization is destroyed to save memory
        - Cannot call solve(), logdet(), or selected_inverse() again
        - Solver is invalidated (_is_factorized becomes False)
        - Use only as final operation with current factors

        Examples
        --------
        Safe usage (keeps solver usable):
        >>> inv = solver.selected_inverse()
        >>> x = solver.solve(b)  # Still works

        Memory-efficient usage (solver becomes unusable before next call to update_matrix()):
        >>> inv = solver.selected_inverse(overwrite_factors=True)
        >>> solver.solve(b)  # ERROR: not factorized
        """
        if not self._is_factorized:
            raise RuntimeError(
                "Matrix must be factorized before performing selected inversion"
            )

        result: Matrix = self._compute_selected_inverse(
            overwrite_factors=overwrite_factors
        )

        if overwrite_factors:
            # Invalidate solver state - factors were destroyed
            self._is_factorized: bool = False
            self._factors: bool = None

        return result

    # 11. Private/protected methods (start with _)
    @abstractmethod
    def _compute_factorization(self, overwrite: bool = False):
        """Compute factorization, optionally in-place.

        Parameters
        ----------
        overwrite : bool
            If True, may overwrite self._matrix._data to save memory.

        Returns
        -------
        factorization : object
            Factorization data structure (method-specific).
        """

    @abstractmethod
    def _solve_system(self, b):
        """Solve linear system using stored factorization.

        Parameters
        ----------
        b : numpy.ndarray
            Right-hand side (unwrapped from Matrix if needed).

        Returns
        -------
        x : numpy.ndarray
            Solution vector.
        """

    @abstractmethod
    def _compute_logdet(self):
        """Compute log-determinant from factorization."""

    @abstractmethod
    def _compute_selected_inverse(self, overwrite_factors: bool = False) -> Matrix:
        """Perform selected inversion using factorization.

        Parameters
        ----------
        overwrite_factors : bool
            If True, destroys stored factorization to save memory.
        """

    def _choose_target(self):
        """Choose hardware target based on matrix type."""
        return self._matrix.hw_target
