from dalia.core.gradient_method import GradientMethod
from dalia import xp

from dalia.utils import get_device


class VanillaGradient(GradientMethod):
    """Vanilla finite difference gradient computation method."""

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
        super().__init__(
            basis_size=basis_size, finite_difference_epsilon=finite_difference_epsilon
        )

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
        theta_dev = get_device(theta)
        direction_matrix.fill(0.0)

        # First column is the current theta
        direction_matrix[:, 0] = theta_dev
        # Next basis_size columns are + epsilon * e_i
        direction_matrix[
            :, 1 : self.basis_size + 1
        ] += self.finite_difference_epsilon * self.basis + xp.repeat(
            theta_dev.reshape(-1, 1), self.basis_size, 1
        )
        # Next basis_size columns are - epsilon * e_i
        direction_matrix[
            :, 1 + self.basis_size :
        ] -= self.finite_difference_epsilon * self.basis - xp.repeat(
            theta_dev.reshape(-1, 1), self.basis_size, 1
        )

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
        for i in range(self.basis_size):
            gradient[i] = (
                function_evaluations[i + 1]
                - function_evaluations[self.basis_size + i + 1]
            ) / (2.0 * self.finite_difference_epsilon)
