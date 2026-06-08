# tests/backend/units/linalg/solvers/test_solve.py

import pytest
import numpy as np

from dalia.backend.linalg.solvers import DenseSolver, SparseSolver, CuDSS
from dalia.backend.config import memory_regime, cupy_version, nvmath_version

from .conftest import INTERNAL_DEVICE_TYPES, INTERNAL_MATRIX_TYPES

class TestSolve:

    @pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
    def test_solve_dense(self, matrix_factory, device_type):
        """Test solving a dense linear system Ax = b."""
        M = matrix_factory("DenseMatrix", shape=(3, 3), hw_target=device_type)
        E = matrix_factory("DenseMatrix", shape=(3, 3), data=np.eye(3), hw_target=device_type)
        A = M.T @ M + E  # Make it symmetric positive definite
        b = np.array([1, 2, 3], dtype=np.float64)
        solver = DenseSolver(A)
        solver.factorize()
        x = solver.solve(b)
        # Verify the solution is correct
        if device_type == "accelerator":
            x =x.get()
        assert np.allclose(A.toarray() @ x, b)

    def test_solve_sparse(self, matrix_factory, device_type = "host"):
        """Test solving a sparse linear system Ax = b."""
        M = matrix_factory("SparseMatrix", shape=(3, 3), hw_target=device_type)
        E = matrix_factory("SparseMatrix", shape=(3, 3), data=np.eye(3), hw_target=device_type)
        A = M.T @ M + E  # Make it symmetric positive definite
        b = np.array([1, 2, 3], dtype=np.float64)
        solver = SparseSolver(A)
        solver.factorize()
        x = solver.solve(b)
        # Verify the solution is correct
        if device_type == "accelerator":
            x = x.get()
        assert np.allclose(A.toarray() @ x, b)

    def test_solve_cudss(self, matrix_factory):
        """Test solving a linear system using CuDSS."""
        if cupy_version is None:
            pytest.skip("CuPy is not installed")
        if nvmath_version is None:
            pytest.skip("NVMATH is not installed")
        M = matrix_factory("SparseMatrix", shape=(3, 3), hw_target="accelerator")
        E = matrix_factory("SparseMatrix", shape=(3, 3), data=np.eye(3), hw_target="accelerator")
        A = M.T @ M + E  # Make it symmetric positive definite
        b = np.array([1., 2., 3.], dtype=np.float64)
        solver = CuDSS(A)
        x = solver.solve(b)
        # Verify the solution is correct
        x = x.get()
        assert np.allclose(A.toarray() @ x, b)

