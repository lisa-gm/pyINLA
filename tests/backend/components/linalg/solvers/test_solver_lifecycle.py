# tests/backend/components/linalg/solvers/test_solver_lifecycle.py


class TestSolverLifecycle:
    """Test solver state transitions across different workflows."""

    def test_basic_cycle(self):
        """Test workflow:
        -> init
        -> assert(not factorized)
        -> factorize
        -> assert(is factorized)
        -> solve
        -> assert(correctness).
        """

    def test_update_cycle(self):
        """Test workflow:
        -> init
        -> factorize
        -> solve
        -> update_matrix
        -> assert(not factorized)
        -> refactorize
        -> assert(factorized)
        -> solve
        -> assert(correctness).
        """

    def test_overwrite_factors_cycle(self):
        """Test workflow:
        -> init
        -> factorize
        -> solve
        -> selected_inverse(overwrite=True)
        -> assert(invalid)
        -> update_matrix
        -> factorize
        -> assert(factorized).
        """
