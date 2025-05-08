# Copyright 2024-2025 pyINLA authors. All rights reserved.

from pyinla import NDArray, sp, xp
from pyinla.configs.pyinla_config import SolverConfig
from pyinla.core.solver import Solver

## check if sparse matrix is diagonal 
def is_diagonal(A: NDArray) -> bool:
    """Check if a matrix is diagonal."""

    coo = A.tocoo()
    return xp.all(coo.row == coo.col)

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

        if sp.sparse.issparse(A):

            ## TODO: where should this go in the long term?
            if is_diagonal(A):
                # if A is diagonal, we can use the diagonal directly
                self.L[:] = 0
                #self.L.diagonal()[:] = xp.sqrt(A.diagonal())
                self.L[xp.arange(self.n), xp.arange(self.n)] = xp.sqrt(A.diagonal())
                return

            else:    
                self.L[:] = A.todense()
        else:
            ## TODO: can we safely overwrite A?!
            self.L[:] = A

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
    