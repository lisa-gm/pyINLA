# Copyright 2024 pyINLA authors. All rights reserved.

from pyinla import NDArray, sp, xp
from pyinla.configs.pyinla_config import SolverConfig
from pyinla.core.solver import Solver


class DenseSolver(Solver):
    def __init__(
        self,
        solver_config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(solver_config)

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
