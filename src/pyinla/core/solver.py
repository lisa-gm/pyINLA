# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike
from scipy.sparse import sparray

from pyinla.core.pyinla_config import PyinlaConfig


class Solver(ABC):
    """Abstract core class for numerical solvers."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the solver."""
        self.pyinla_config = pyinla_config

    @abstractmethod
    def cholesky(self, A: sparray) -> None:
        """Compute Cholesky factor of input matrix."""
        pass

    @abstractmethod
    def solve(self, rhs: ArrayLike) -> ArrayLike:
        """Solve linear system using Cholesky factor."""
        pass

    @abstractmethod
    def logdet(self) -> float:
        """Compute logdet of input matrix using Cholesky factor."""
        pass

    @abstractmethod
    def full_inverse(self) -> ArrayLike:
        """Compute inverse of input matrix using Cholesky factor."""
        pass

    @abstractmethod
    def selected_inverse(self) -> ArrayLike:
        """Compute inverse of selected rows of input matrix using Cholesky factor."""
        pass
