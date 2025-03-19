# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import NDArray, sp, xp
from pyinla.configs.pyinla_config import SolverConfig
from pyinla.core.solver import Solver


class DenseSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the DenseSolver class.

        Parameters
        ----------
        config : SolverConfig
            Configuration object for the solver.
        n : int
            Size of the matrix.

        Returns
        -------
        None
        """
        super().__init__(config)

        self.n: int = kwargs.get("n", None)
        assert self.n is not None, "The size of the matrix must be provided."

        self.L: NDArray = xp.zeros((self.n, self.n), dtype=xp.float64)

    def cholesky(self, A: NDArray, **kwargs) -> None:
        self.L[:] = A.todense()

        self.L = xp.linalg.cholesky(self.L)

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        sp.linalg.solve_triangular(self.L, rhs, lower=True, overwrite_b=True)
        sp.linalg.solve_triangular(self.L.T, rhs, lower=False, overwrite_b=True)

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        return 2 * xp.sum(xp.log(xp.diag(self.L)))

    # TODO: optimize for memory??
    def selected_inversion(self, A: NDArray, **kwargs) -> None:

        self.L[:] = A.todense()
        self.L = xp.linalg.cholesky(self.L)

        L_inv = sp.linalg.solve_triangular(self.L, xp.eye(self.L.shape[0]), lower=True)
        self.A_inv = L_inv.T @ L_inv

        return self.A_inv


    def _structured_to_spmatrix(self, A: sp.sparse.spmatrix) -> None:

        B = A.tocoo()
        B.data = self.A_inv[B.row, B.col]

        return B.tocsr()
    
from scipy.sparse import random, csc_matrix
import numpy as np
if __name__ == "__main__":

    # Set matrix size (larger matrix = more noticeable multithreading impact)
    n = 6

    # Generate a random sparse symmetric positive-definite matrix
    A = random(n, n, density=0.2, format="csc")

    # Make the matrix symmetric
    A = (A + A.T) / 2

    # Add n * I to ensure positive definiteness
    A += n * csc_matrix(np.eye(n))
    A_dense = A.todense()

    # sparsity pattern of A
    # Create a SolverConfig instance
    config = SolverConfig()

    # compute the inverse of A
    solver = DenseSolver(config, n=n)
    A_inv = solver.selected_inversion(A)
    print(A_inv)
    print(np.linalg.inv(A_dense))
    print(np.allclose(A_inv, np.linalg.inv(A_dense)))

    A_inv_spmatrix = solver._structured_to_spmatrix(A)
    print(A_inv_spmatrix.toarray())

    print(A_inv)

    diff = A_inv_spmatrix - A_inv
    print("diff: ", diff)
    
    # convert the structured matrix to a sparse matrix

