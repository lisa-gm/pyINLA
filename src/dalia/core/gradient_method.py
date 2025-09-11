from abc import ABC, abstractmethod

from dalia import xp


class GradientMethod(ABC):
    """Core class for gradient computation methods."""

    def __init__(self, basis_size, finite_difference_epsilon) -> None:
        """Initialize the gradient computation method.

        Parameters
        ----------
        basis_size : int
            The size of the basis for finite differences.
        finite_difference_epsilon : float
            The epsilon value for finite difference computations.

        Returns
        -------
        None
        """
        self.basis_size = basis_size
        self.basis = xp.identity(self.basis_size, dtype=xp.float64)
        self.finite_difference_epsilon = finite_difference_epsilon

    @abstractmethod
    def get_evaluation_directions(self, direction_matrix, theta) -> None:
        """Get the evaluation directions for the gradient computation.

        Parameters
        ----------
        direction_matrix : xp.ndarray
            The matrix to store the evaluation directions.
        theta : xp.ndarray
            The current (hyper)parameter values.

        Returns
        -------
        None
        """
        ...

    @abstractmethod
    def compute_gradient(self, function_evaluations, gradient) -> None:
        """Compute the gradient using finite differences.

        Parameters
        ----------
        function_evaluations : xp.ndarray
            The function evaluations at the current and perturbed points.
        gradient : xp.ndarray
            The array to store the computed gradient.

        Returns
        -------
        None
        """
        ...
