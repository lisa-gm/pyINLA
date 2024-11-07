# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from scipy.sparse import sparray

from pyinla import ArrayLike
from pyinla.core.pyinla_config import PyinlaConfig


class Solver(ABC):
    """Abstract core class for numerical solvers."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        self.pyinla_config = pyinla_config

    @abstractmethod
    def cholesky(self, A: sparray, **kwargs) -> None:
        """Compute Cholesky factor of input matrix."""
        pass

    @abstractmethod
    def solve(self, rhs: ArrayLike, **kwargs) -> ArrayLike:
        """Solve linear system using Cholesky factor."""
        pass

    @abstractmethod
    def logdet(self, **kwargs) -> float:
        """Compute logdet of input matrix using Cholesky factor."""
        pass

    @abstractmethod
    def full_inverse(self, **kwargs) -> ArrayLike:
        """Compute inverse of input matrix using Cholesky factor."""
        pass

    @abstractmethod
    def selected_inverse(self, **kwargs) -> ArrayLike:
        """Compute inverse of selected rows of input matrix using Cholesky factor."""
        pass

    @abstractmethod
    def get_L(self, **kwargs) -> ArrayLike:
        """Get L as a dense array."""
        pass
