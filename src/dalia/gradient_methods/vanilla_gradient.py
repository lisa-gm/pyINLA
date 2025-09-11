from dalia.core.gradient_method import GradientMethod
from dalia import xp

from dalia.utils import get_device


class VanillaGradient(GradientMethod):
    """Vanilla finite difference gradient computation method."""

    def __init__(self, basis_size, finite_difference_epsilon):
        super().__init__(
            basis_size=basis_size, finite_difference_epsilon=finite_difference_epsilon
        )

    def get_evaluation_directions(self, direction_matrix, theta) -> None:
        direction_matrix.fill(0.0)
        direction_matrix[:, 0] = get_device(theta)

        print(f"theta: {theta}")

        direction_matrix[
            :, 1 : self.basis.shape[0] + 1
        ] += self.finite_difference_epsilon * self.basis + xp.repeat(
            get_device(theta).reshape(-1, 1), self.basis.shape[0], 1
        )
        direction_matrix[
            :, 1 + self.basis.shape[0] :
        ] -= self.finite_difference_epsilon * self.basis - xp.repeat(
            get_device(theta).reshape(-1, 1), self.basis.shape[0], 1
        )

    def compute_gradient(self, function_evaluations, gradient) -> None:

        print(f"IG: function_evaluations: {function_evaluations}")
        for i in range(self.basis.shape[0]):
            gradient[i] = (
                function_evaluations[i + 1]
                - function_evaluations[self.basis.shape[0] + i + 1]
            ) / (2.0 * self.finite_difference_epsilon)
