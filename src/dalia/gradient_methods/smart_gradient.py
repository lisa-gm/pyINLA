from numpy import random as rand

from dalia import xp

from dalia.core.gradient_method import GradientMethod
from dalia.utils import get_device


class SmartGradient(GradientMethod):
    """Smart gradient computation method.

    This method adaptively updates the basis used for finite difference
    gradient estimation based on the changes in the (hyper)parameter values.
    It employs QR decomposition to maintain an orthogonal basis and scales
    the updates to ensure numerical stability.

    References
    ----------
    .. [1] Esmail Abdul Fattah, Janet Van Niekerk, Håvard Rue.
        Smart Gradient - An adaptive technique for improving
        gradient estimation. Foundations of Data Science, 2022,
        4(1): 123-136. doi: 10.3934/fods.2021037
    """

    def __init__(
        self,
        basis_size,
        finite_difference_epsilon,
        diagonal_noise=1e-8,
        scaling_threshold=1e-12,
    ) -> None:
        """Initialize the gradient computation method.

        Parameters
        ----------
        basis_size : int
            The size of the basis for finite differences.
        finite_difference_epsilon : float
            The epsilon value for finite difference computations.
        diagonal_noise : float
            The diagonal noise to avoid singularities in the QR decomposition.
        scaling_threshold : float
            The threshold below which scaling is not performed.

        Returns
        -------
        None
        """
        super().__init__(
            basis_size=basis_size, finite_difference_epsilon=finite_difference_epsilon
        )
        self.diagonal_noise = diagonal_noise
        self.scaling_threshold = scaling_threshold
        self.temp_basis = xp.identity(self.basis_size)
        self.prev_theta = xp.zeros(self.basis_size)
        self.curr_theta = xp.zeros(self.basis_size)
        self.count = 0
        self.rng = rand.default_rng()

    def _transformed_fun(self, phi) -> xp.ndarray:
        """Transform the input using the current theta and basis.

        Parameters
        ----------
        phi : xp.ndarray
            The input to transform.

        Returns
        -------
        xp.ndarray
            The transformed output.
        """
        return self.curr_theta + self.basis @ phi

    def _scale(self, x) -> xp.ndarray:
        """Scale the input vector.

        Parameters
        ----------
        x : xp.ndarray
            The input vector to scale.

        Returns
        -------
        xp.ndarray
            The scaled output vector.
        """
        mean = xp.mean(x)
        std = xp.std(x, ddof=1)
        if std < self.scaling_threshold:
            return x - mean
        return (x - mean) / std

    def _update_basis(self, current_theta) -> None:
        """Update the basis using the current theta.

        Parameters
        ----------
        current_theta : xp.ndarray
            The current (hyper)parameter values.

        Returns
        -------
        None
        """
        self.curr_theta = current_theta
        self.temp_basis = xp.roll(self.temp_basis, 1, axis=1)
        xdiff = current_theta - self.prev_theta
        xdiff += get_device(self.rng.normal(0.0, self.diagonal_noise, self.basis_size))
        self.temp_basis[:, 0] = self._scale(xdiff)
        self.basis = self.temp_basis

    def _orthogonalize_basis(self) -> None:
        """Orthogonalize the basis using QR decomposition.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            Q, R = xp.linalg.qr(self.basis)
            self.basis = Q
        except xp.linalg.LinAlgError:
            print("Warning: QR decomposition failed. Resetting G to identity.")
            self.basis = xp.identity(self.basis_size)

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

        if self.count > 0:
            self._update_basis(theta_dev)
        else:
            self.curr_theta = theta_dev  # Set the starting point

        self._orthogonalize_basis()

        self.prev_theta = xp.copy(theta_dev)
        self.count += 1

        # First column is the current theta
        direction_matrix[:, 0] = theta_dev
        # Next basis_size columns are + epsilon * e_i
        direction_matrix[:, 1 : 1 + self.basis_size] += (
            self.finite_difference_epsilon * self.basis
        )
        # Next basis_size columns are - epsilon * e_i
        direction_matrix[:, self.basis_size + 1 :] -= (
            self.finite_difference_epsilon * self.basis
        )

        for i in range(1, direction_matrix.shape[1]):
            direction_matrix[:, i] = self._transformed_fun(phi=direction_matrix[:, i]).T

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

        # Transform back the gradient into the original basis
        gradient[:] = xp.linalg.solve(self.basis.T, gradient)
