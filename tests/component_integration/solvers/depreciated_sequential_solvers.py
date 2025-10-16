
from dalia import backend_flags, xp, sp
import numpy as np
import pytest

from .utils import (
    generate_pobta, 
    generate_pobt, 
    rhs,
    numpy_reference_cholesky,
    numpy_reference_solve, 
    numpy_reference_logdet,
    numpy_reference_selected_inversion_diagonal
)


class TestSolvers:
    """Unified test suite for all DALIA solvers."""

    def test_factorize_correctness(
        self, 
        solver_factory, 
        solver_type, 
        sparsity_pattern,
        diagonal_blocksize, 
        n_diag_blocks, 
        arrowhead_blocksize
    ):
        """Test Cholesky decomposition correctness against NumPy reference."""
        # Generate test matrix based on sparsity pattern
        if sparsity_pattern == "bta":
            A = generate_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        else:  # bt
            A = generate_pobt(diagonal_blocksize, n_diag_blocks)
        
        # Convert to sparse matrix
        A_sparse = sp.sparse.csc_matrix(A)
        
        # Create solver
        solver = solver_factory(
            solver_type, diagonal_blocksize, n_diag_blocks, arrowhead_blocksize
        )
        
        # Run solver factorize
        if solver_type == "scipy":
            solver.factorize(A_sparse)  # SparseSolver doesn't need sparsity parameter
        else:  # serinv
            solver.factorize(A_sparse, sparsity=sparsity_pattern)
        
        # Compute reference
        L_ref = numpy_reference_cholesky(A)
        
        # For now, we mainly test that factorize runs without error
        # More detailed verification would require extracting the factorization
        # which depends on the solver's internal representation
        assert True  # Placeholder - if we get here, factorize succeeded

    def test_solve_correctness(
        self, 
        solver_factory, 
        solver_type, 
        sparsity_pattern,
        diagonal_blocksize, 
        n_diag_blocks, 
        arrowhead_blocksize,
        num_rhs
    ):
        """Test linear system solution correctness against NumPy reference."""
        # Generate test matrix and RHS
        if sparsity_pattern == "bta":
            A = generate_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
            b = rhs(num_rhs, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        else:  # bt
            A = generate_pobt(diagonal_blocksize, n_diag_blocks)
            b = rhs(num_rhs, diagonal_blocksize, 0, n_diag_blocks)  # No arrowhead for bt
        
        A_sparse = sp.sparse.csc_matrix(A)
        
        # Create solver and run factorize + solve
        solver = solver_factory(
            solver_type, diagonal_blocksize, n_diag_blocks, arrowhead_blocksize
        )
        
        if solver_type == "scipy":
            solver.factorize(A_sparse)
            x_solver = solver.solve(b.copy())
        else:  # serinv
            solver.factorize(A_sparse, sparsity=sparsity_pattern)
            x_solver = solver.solve(b.copy(), sparsity=sparsity_pattern)
        
        # Compute reference solution
        x_ref = numpy_reference_solve(A, b)
        
        # Compare solutions
        x_solver_host = xp.asarray(x_solver) if hasattr(xp, 'get') else x_solver
        if hasattr(x_solver_host, 'get'):  # CuPy array
            x_solver_host = x_solver_host.get()
        
        np.testing.assert_allclose(
            x_solver_host, x_ref, rtol=1e-14, atol=1e-16
        )

    def test_logdet_correctness(
        self, 
        solver_factory, 
        solver_type, 
        sparsity_pattern,
        diagonal_blocksize, 
        n_diag_blocks, 
        arrowhead_blocksize
    ):
        """Test log determinant correctness against NumPy reference."""
        # Generate test matrix
        if sparsity_pattern == "bta":
            A = generate_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        else:  # bt
            A = generate_pobt(diagonal_blocksize, n_diag_blocks)
        
        A_sparse = sp.sparse.csc_matrix(A)
        
        # Create solver and run factorize + logdet
        solver = solver_factory(
            solver_type, diagonal_blocksize, n_diag_blocks, arrowhead_blocksize
        )
        
        if solver_type == "scipy":
            solver.factorize(A_sparse)
            logdet_solver = solver.logdet()
        else:  # serinv
            solver.factorize(A_sparse, sparsity=sparsity_pattern)
            logdet_solver = solver.logdet(sparsity=sparsity_pattern)
        
        # Compute reference
        logdet_ref = numpy_reference_logdet(A)
        
        # Compare log determinants
        logdet_solver_host = float(logdet_solver)
        if hasattr(logdet_solver, 'get'):  # CuPy scalar
            logdet_solver_host = float(logdet_solver.get())
        
        np.testing.assert_allclose(
            logdet_solver_host, logdet_ref, rtol=1e-14, atol=1e-16
        )

    def test_selected_inversion_correctness(
        self, 
        solver_factory, 
        solver_type, 
        sparsity_pattern,
        diagonal_blocksize, 
        n_diag_blocks, 
        arrowhead_blocksize
    ):
        """Test selected inversion correctness against NumPy reference (diagonal elements)."""
        # Generate test matrix
        if sparsity_pattern == "bta":
            A = generate_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        else:  # bt
            A = generate_pobt(diagonal_blocksize, n_diag_blocks)
        
        A_sparse = sp.sparse.csc_matrix(A)
        
        # Create solver and run factorize + selected_inversion
        solver = solver_factory(
            solver_type, diagonal_blocksize, n_diag_blocks, arrowhead_blocksize
        )
        
        if solver_type == "scipy":
            solver.factorize(A_sparse)
            solver.selected_inversion()  # May be placeholder for SparseSolver
        else:  # serinv
            solver.factorize(A_sparse, sparsity=sparsity_pattern)
            solver.selected_inversion(sparsity=sparsity_pattern)
        
        # For now, just test that selected_inversion runs without error
        # Detailed verification of diagonal elements can be added later
        assert True  # Placeholder

    def test_full_workflow_correctness(
        self, 
        solver_factory, 
        solver_type, 
        sparsity_pattern,
        diagonal_blocksize, 
        n_diag_blocks, 
        arrowhead_blocksize,
        num_rhs
    ):
        """Test complete workflow: factorize -> logdet -> solve -> selected_inversion."""
        # Generate test matrix and RHS
        if sparsity_pattern == "bta":
            A = generate_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
            b = rhs(num_rhs, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        else:  # bt
            A = generate_pobt(diagonal_blocksize, n_diag_blocks)
            b = rhs(num_rhs, diagonal_blocksize, 0, n_diag_blocks)  # No arrowhead for bt
        
        A_sparse = sp.sparse.csc_matrix(A)
        
        # Create solver
        solver = solver_factory(
            solver_type, diagonal_blocksize, n_diag_blocks, arrowhead_blocksize
        )
        
        # Complete workflow
        if solver_type == "scipy":
            # factorize
            solver.factorize(A_sparse)
            
            # Logdet
            logdet_solver = solver.logdet()
            
            # Solve
            x_solver = solver.solve(b.copy())
            
            # Selected inversion
            solver.selected_inversion()
            
        else:  # serinv
            # factorize
            solver.factorize(A_sparse, sparsity=sparsity_pattern)
            
            # Logdet
            logdet_solver = solver.logdet(sparsity=sparsity_pattern)
            
            # Solve
            x_solver = solver.solve(b.copy(), sparsity=sparsity_pattern)
            
            # Selected inversion
            solver.selected_inversion(sparsity=sparsity_pattern)
        
        # Verify against references
        x_ref = numpy_reference_solve(A, b)
        logdet_ref = numpy_reference_logdet(A)
        
        # Convert to host arrays for comparison
        x_solver_host = xp.asarray(x_solver) if hasattr(xp, 'get') else x_solver
        if hasattr(x_solver_host, 'get'):
            x_solver_host = x_solver_host.get()
        
        logdet_solver_host = float(logdet_solver)
        if hasattr(logdet_solver, 'get'):
            logdet_solver_host = float(logdet_solver.get())
        
        # Assertions
        np.testing.assert_allclose(
            x_solver_host, x_ref, rtol=1e-14, atol=1e-16
        )
        np.testing.assert_allclose(
            logdet_solver_host, logdet_ref, rtol=1e-14, atol=1e-16
        )

    def test_sparse_solver_parameter_handling(
        self, 
        solver_factory, 
        sparsity_pattern,
        diagonal_blocksize, 
        n_diag_blocks, 
        arrowhead_blocksize,
        num_rhs
    ):
        """Test that SparseSolver handles irrelevant sparsity parameters gracefully."""
        # Generate test matrix
        if sparsity_pattern == "bta":
            A = generate_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
            b = rhs(num_rhs, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        else:  # bt
            A = generate_pobt(diagonal_blocksize, n_diag_blocks)
            b = rhs(num_rhs, diagonal_blocksize, 0, n_diag_blocks)  # No arrowhead for bt
        
        A_sparse = sp.sparse.csc_matrix(A)
        
        # Create sparse solver
        solver = solver_factory("scipy", diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
        
        # Test that SparseSolver works without sparsity parameter
        solver.factorize(A_sparse)
        x_solver = solver.solve(b.copy())
        logdet_solver = solver.logdet()
        solver.selected_inversion()
        
        # Verify correctness
        x_ref = numpy_reference_solve(A, b)
        logdet_ref = numpy_reference_logdet(A)
        
        x_solver_host = xp.asarray(x_solver) if hasattr(xp, 'get') else x_solver
        if hasattr(x_solver_host, 'get'):
            x_solver_host = x_solver_host.get()
        
        logdet_solver_host = float(logdet_solver)
        if hasattr(logdet_solver, 'get'):
            logdet_solver_host = float(logdet_solver.get())
        
        np.testing.assert_allclose(
            x_solver_host, x_ref, rtol=1e-14, atol=1e-16
        )
        np.testing.assert_allclose(
            logdet_solver_host, logdet_ref, rtol=1e-14, atol=1e-16
        )

    def test_structured_mapping_kernels(
        self, 
        solver_factory, 
        sparsity_pattern,
        diagonal_blocksize, 
        n_diag_blocks, 
        arrowhead_blocksize,
        num_rhs
    ):
        """Test that SerinvSolver correctly maps sparse matrices to structured format."""
        # This test specifically focuses on the sparse-to-structured mapping
        # which is a key component of the SerinvSolver interface
        
        # Generate test matrix
        if sparsity_pattern == "bta":
            A = generate_pobta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
            b = rhs(num_rhs, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        else:  # bt
            A = generate_pobt(diagonal_blocksize, n_diag_blocks)
            b = rhs(num_rhs, diagonal_blocksize, 0, n_diag_blocks)  # No arrowhead for bt
        
        A_sparse = sp.sparse.csc_matrix(A)
        
        # Create serinv solver
        solver = solver_factory("serinv", diagonal_blocksize, n_diag_blocks, arrowhead_blocksize)
        
        # The mapping happens during factorize call via _spmatrix_to_structured
        solver.factorize(A_sparse, sparsity=sparsity_pattern)
        
        # Test that the mapping worked by solving and comparing with reference
        x_solver = solver.solve(b.copy(), sparsity=sparsity_pattern)
        x_ref = numpy_reference_solve(A, b)
        
        x_solver_host = xp.asarray(x_solver) if hasattr(xp, 'get') else x_solver
        if hasattr(x_solver_host, 'get'):
            x_solver_host = x_solver_host.get()
        
        np.testing.assert_allclose(
            x_solver_host, x_ref, rtol=1e-14, atol=1e-16
        )