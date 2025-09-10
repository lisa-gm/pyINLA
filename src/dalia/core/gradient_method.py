from abc import ABC, abstractmethod

from dalia import xp


class GradientMethod(ABC):
    """Core class for gradient computation methods."""

    def __init__(self, basis_size, finite_difference_epsilon):
        self.basis = xp.identity((basis_size, basis_size), dtype=xp.float64)
        self.finite_difference_epsilon = finite_difference_epsilon

    @abstractmethod
    def get_evaluation_directions(self, direction_matrix) -> None:
        ...

    @abstractmethod
    def compute_gradient(self, gradient) -> None:
        ...