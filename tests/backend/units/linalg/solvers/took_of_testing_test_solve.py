# tests/backend/units/linalg/solvers/test_solve.py

import numpy as np
import pytest

from dalia.backend.config import cupy_version, memory_regime, nvmath_version
from dalia.backend.linalg.solvers import CuDSS, DenseSolver, SparseSolver

if cupy_version is not None:
    import cupy as cp

from .conftest import INTERNAL_DEVICE_TYPES, INTERNAL_MATRIX_TYPES


class TestSolve:

    @pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
    def test_solve_dense(self, matrix_factory, device_type):
        """Test solving a dense linear system Ax = b."""
        M = matrix_factory("DenseMatrix", shape=(3, 3), hw_target=device_type)
        E = matrix_factory(
            "DenseMatrix", shape=(3, 3), data=np.eye(3), hw_target=device_type
        )
        A = M.T @ M + E  # Make it symmetric positive definite
        b = np.array([1, 2, 3], dtype=np.float64)
        if device_type == "accelerator":
            b = cp.asarray(b)
        solver = DenseSolver(A)
        solver.factorize()
        x = solver.solve(b)
        # Verify the solution is correct
        if device_type == "accelerator":
            x = x.get()
            b = b.get()
        assert np.allclose(A.toarray() @ x, b)

    @pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
    def test_solve_sparse(self, matrix_factory, device_type):
        """Test solving a sparse linear system Ax = b."""
        M = matrix_factory("SparseMatrix", shape=(3, 3), hw_target=device_type)
        E = matrix_factory(
            "SparseMatrix", shape=(3, 3), data=np.eye(3), hw_target=device_type
        )
        A = M.T @ M + E  # Make it symmetric positive definite
        b = np.array([1, 2, 3], dtype=np.float64)
        if device_type == "accelerator":
            b = cp.asarray(b)
        solver = SparseSolver(A)
        solver.factorize()
        x = solver.solve(b)
        # Verify the solution is correct
        if device_type == "accelerator":
            x = x.get()
            b = b.get()
        assert np.allclose(A.toarray() @ x, b)

    def test_solve_cudss(self, matrix_factory):
        """Test solving a linear system using CuDSS."""
        if cupy_version is None:
            pytest.skip("CuPy is not installed")
        if nvmath_version is None:
            pytest.skip("NVMATH is not installed")
        M = matrix_factory("SparseMatrix", shape=(3, 3), hw_target="accelerator")
        E = matrix_factory(
            "SparseMatrix", shape=(3, 3), data=np.eye(3), hw_target="accelerator"
        )
        A = M.T @ M + E  # Make it symmetric positive definite
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = cp.asarray(b)
        solver = CuDSS(A)
        x = solver.solve(b)
        # Verify the solution is correct
        x = x.get()
        b = b.get()
        assert np.allclose(A.toarray() @ x, b)
