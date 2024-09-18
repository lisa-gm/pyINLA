# Copyright 2024 pyINLA authors. All rights reserved.

from numpy.typing import ArrayLike

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.core.solver import Solver


class SerinvSolver(Solver):
    """Serinv Solver class."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the solver."""
        self.pyinla_config = pyinla_config

    def cholesky(self) -> None:
        """Compute Cholesky factor of input matrix."""
        pass

    def solve(self) -> ArrayLike:
        """Solve linear system using Cholesky factor."""
        pass

    def logdet(self) -> float:
        """Compute logdet of input matrix using cholesky factor."""
        pass
