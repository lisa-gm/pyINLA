from dalia.core.gradient_method import GradientMethod
from dalia import xp



class VanillaGradient(GradientMethod):
    """Vanilla finite difference gradient computation method."""

    def __init__(self, basis_size, finite_difference_epsilon):
        super().__init__(basis_size=basis_size, finite_difference_epsilon=finite_difference_epsilon)

    def get_evaluation_directions(self, direction_matrix) -> None:
        direction_matrix[:, 1 : self.basis.shape[0] + 1] += self.finite_difference_epsilon * self.basis
        direction_matrix[:, 1 + self.basis.shape[0]:] -= self.finite_difference_epsilon * self.basis

        print(direction_matrix)

    def compute_gradient(self, gradient) -> None:
        for i in range(self.model.n_hyperparameters):
            gradient[i] = (self.f_evaluations[2 * i + 1] - self.f_evaluations[2 * i]) / (
                2.0 * self.h
            )