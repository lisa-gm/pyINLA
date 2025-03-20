# Copyright 2024-2025 pyINLA authors. All rights reserved.

import numpy as np
from scipy.sparse import random

from pyinla import NDArray, sp, spl, xp
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
        rhs[:] = sp.linalg.solve_triangular(self.L, rhs, lower=True, overwrite_b=True)
        rhs[:] = sp.linalg.solve_triangular(
            self.L.T, rhs, lower=False, overwrite_b=True
        )

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

        L_inv = xp.eye(self.L.shape[0])
        L_inv[:] = spl.solve_triangular(self.L, L_inv, lower=True, overwrite_b=True)
        self.A_inv = L_inv.T @ L_inv

        return self.A_inv

    def _structured_to_spmatrix(self, A: sp.sparse.spmatrix) -> None:
        B = A.tocoo()
        B.data = self.A_inv[B.row, B.col]

        return B


if __name__ == "__main__":
    # Set matrix size (larger matrix = more noticeable multithreading impact)
    n = 10

    # Generate a random sparse symmetric positive-definite matrix
    A = random(n, n, density=0.2, format="csc")

    if xp is not np:
        # copy sparse matrix a to GPU
        A = sp.sparse.csc_matrix(A)

    # Make the matrix symmetric
    A = (A + A.T) / 2

    # Add n * I to ensure positive definiteness
    A += n * sp.sparse.csc_matrix(xp.eye(n))
    A_dense = A.todense()
    A_inv_ref = xp.linalg.inv(A_dense)
    # sparsity pattern of A
    # Create a SolverConfig instance
    config = SolverConfig()

    # compute the inverse of A
    solver = DenseSolver(config, n=n)
    A_inv = solver.selected_inversion(A)
    # print("A: \n", A_dense)
    # print("A_inv: \n", A_inv)
    # print("A_inv_ref: \n", A_inv_ref)
    # print(np.linalg.inv(A_dense))
    print(np.allclose(A_inv, A_inv_ref))

    A_inv_spmatrix = solver._structured_to_spmatrix(A)
    # print(A_inv_spmatrix.toarray())

    # print(A_inv)

    # diff = A_inv - xp.linalg.inv(A_dense)
    # print("diff: ", diff)

    # convert the structured matrix to a sparse matrix
