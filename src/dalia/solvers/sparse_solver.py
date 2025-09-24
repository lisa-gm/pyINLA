# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import NDArray, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver


class SparseSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config)

        self.L: sp.sparse.spmatrix = None

    def cholesky(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        """Compute Cholesky factor of input matrix."""

        A = sp.sparse.csc_matrix(A)

        LU = sp.sparse.linalg.splu(A, diag_pivot_thresh=0, permc_spec="NATURAL")

        if (LU.U.diagonal() > 0).all():  # Check the matrix A is positive definite.
            self.L = LU.L.dot(sp.sparse.diags(LU.U.diagonal() ** 0.5))
        else:
            print("min(diag(L)): ", xp.min(LU.U.diagonal()))
            raise ValueError("The matrix is not positive definite")

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        rhs[:] = sp.sparse.linalg.spsolve_triangular(
            self.L, rhs, lower=True, overwrite_b=True
        )
        rhs[:] = sp.sparse.linalg.spsolve_triangular(
            self.L.T, rhs, lower=False, overwrite_b=True
        )

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        return 2 * xp.sum(xp.log(self.L.diagonal()))

    def selected_inversion(self, **kwargs) -> None:
        # convert to dense
        L_dense = self.L.toarray()
        L_inv = xp.eye(self.L.shape[0])

        L_inv[:] = sp.linalg.solve_triangular(
            L_dense, L_inv, lower=True, overwrite_b=True
        )
        self.A_inv = L_inv.T @ L_inv

        return self.A_inv

    def _structured_to_spmatrix(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        B = A.tocoo()
        B.data = self.A_inv[B.row, B.col]

        return B

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""
        if self.L is None:
            return 0

        return self.L.data.nbytes + self.L.indptr.nbytes + self.L.indices.nbytes


def main():
    """Test function for SparseSolver."""
    import numpy as np
    from dalia.configs.dalia_config import SolverConfig

    print("=== Testing SparseSolver ===")

    # Create a simple SPD test matrix (tridiagonal)
    n = 10
    A = sp.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format="csr")
    A = -A  # Make it positive definite
    A = A + 3 * sp.sparse.eye(n)  # Ensure diagonal dominance

    print(f"Test matrix shape: {A.shape}")
    print(f"Test matrix nnz: {A.nnz}")
    print("Test matrix structure:")
    print(A.toarray())

    # Create solver config
    config = SolverConfig(type="scipy")
    solver = SparseSolver(config)

    # Test 1: Cholesky decomposition
    print("\n--- Test 1: Cholesky Decomposition ---")
    try:
        solver.cholesky(A)
        print("✓ Cholesky decomposition successful")
        print(f"L factor shape: {solver.L.shape}")
        print(f"L factor nnz: {solver.L.nnz}")

        # Verify L * L^T = A
        reconstructed = solver.L @ solver.L.T
        error = np.linalg.norm((A - reconstructed).toarray())
        print(f"Reconstruction error ||A - L*L^T||: {error:.2e}")

    except Exception as e:
        print(f"✗ Cholesky decomposition failed: {e}")
        return

    # Test 2: Linear solve
    print("\n--- Test 2: Linear System Solve ---")
    try:
        # Create test RHS
        b = np.ones(n)
        b_original = b.copy()

        # Solve Ax = b
        x = solver.solve(b)
        print("✓ Linear solve successful")

        # Verify solution
        residual = np.linalg.norm(A @ x - b_original)
        print(f"Residual ||Ax - b||: {residual:.2e}")
        if residual > 1e-7:
            raise ValueError("Residual too high")

    except Exception as e:
        print(f"✗ Linear solve failed: {e}")
        raise ValueError("Linear solve failed")

    # Test 3: Log determinant
    print("\n--- Test 3: Log Determinant ---")
    try:
        logdet_sparse = solver.logdet()

        # Compare with dense computation
        logdet_dense = np.linalg.slogdet(A.toarray())[1]
        error = abs(logdet_sparse - logdet_dense)

        print(f"Sparse logdet: {logdet_sparse:.6f}")
        print(f"Dense logdet:  {logdet_dense:.6f}")
        print(f"Error: {error:.2e}")
        print("✓ Log determinant computation successful")

    except Exception as e:
        print(f"✗ Log determinant failed: {e}")

    # Test 4: Selected inversion
    print("\n--- Test 4: Selected Inversion ---")
    try:
        A_inv = solver.selected_inversion()
        print("✓ Selected inversion successful")
        print(f"Inverse matrix shape: {A_inv.shape}")

        # Verify A * A_inv = I
        identity = A.toarray() @ A_inv
        identity_error = np.linalg.norm(identity - np.eye(n))
        print(f"Identity error ||A * A_inv - I||: {identity_error:.2e}")

    except Exception as e:
        print(f"✗ Selected inversion failed: {e}")

    # Test 5: Structured matrix operations
    print("\n--- Test 5: Structured Matrix Operations ---")
    try:
        # Create a structured sparse matrix pattern
        pattern = A.copy()
        structured_inv = solver._structured_to_spmatrix(pattern)

        print("solver.A_inv:\n", np.round(solver.A_inv, 4))
        print("structured_inv:\n", structured_inv.toarray())

        print("✓ Structured matrix operation successful")
        print(f"Structured inverse nnz: {structured_inv.nnz}")

    except Exception as e:
        print(f"✗ Structured matrix operation failed: {e}")

    # Test 6: Memory usage
    print("\n--- Test 6: Memory Usage ---")
    try:
        memory_bytes = solver.get_solver_memory()
        memory_mb = memory_bytes / (1024**2)
        print(f"Solver memory usage: {memory_bytes} bytes ({memory_mb:.2f} MB)")
        print("✓ Memory usage computation successful")

    except Exception as e:
        print(f"✗ Memory usage computation failed: {e}")

    # Test 7: Spy plot visualization
    print("\n--- Test 7: Sparsity Pattern Visualization ---")
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original matrix
        axes[0].spy(A, markersize=2)
        axes[0].set_title("Original matrix A")

        # Cholesky factor
        axes[1].spy(solver.L, markersize=2)
        axes[1].set_title("Cholesky factor L")

        # Inverse (if computed)
        if hasattr(solver, "A_inv"):
            axes[2].imshow(solver.A_inv, cmap="viridis")
            axes[2].set_title("Inverse matrix")

        plt.tight_layout()
        plt.savefig("sparse_solver_test.png", dpi=150, bbox_inches="tight")
        print("✓ Sparsity pattern visualization saved as 'sparse_solver_test.png'")

    except ImportError:
        print("⚠ Matplotlib not available for visualization")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")

    print("\n=== SparseSolver Testing Complete ===")


if __name__ == "__main__":
    main()
