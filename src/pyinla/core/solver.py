# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from pyinla import ArrayLike, NDArray
from pyinla.configs.pyinla_config import SolverConfig


class Solver(ABC):
    """Abstract core class for numerical solvers."""

    def __init__(
        self,
        solver_config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        self.solver_config = solver_config

    @abstractmethod
    def cholesky(self, A: ArrayLike, **kwargs) -> None:
        """Compute Cholesky factor of input matrix."""
        pass

    @abstractmethod
    def solve(self, rhs: NDArray, **kwargs) -> NDArray:
        """Solve linear system using Cholesky factor."""
        pass

    @abstractmethod
    def logdet(self, **kwargs) -> float:
        """Compute logdet of input matrix using Cholesky factor."""
        pass
