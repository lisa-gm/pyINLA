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
        self.A_inv = None

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
    def selected_inversion(self, **kwargs) -> None:

        L_inv = xp.eye(self.L.shape[0])
        L_inv[:] = sp.linalg.solve_triangular(
            self.L, L_inv, lower=True, overwrite_b=True
        )
        self.A_inv = L_inv.T @ L_inv

        return self.A_inv

    def _structured_to_spmatrix(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        B = A.tocoo()
        B.data = self.A_inv[B.row, B.col]

        return B

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes."""
        if self.A_inv == None:
            return self.L.nbytes
        else:
            return self.L.nbytes + self.A_inv.nbytes