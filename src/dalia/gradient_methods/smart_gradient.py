from dalia.core.gradient_method import GradientMethod
from dalia import xp


class SmartGradient(GradientMethod):
    """Smart gradient computation method.
    
    
    References
    ----------
    .. [1] Esmail Abdul Fattah, Janet Van Niekerk, Håvard Rue. 
        Smart Gradient - An adaptive technique for improving 
        gradient estimation. Foundations of Data Science, 2022, 
        4(1): 123-136. doi: 10.3934/fods.2021037
    """

    def __init__(self):
        super().__init__()

        # SMART GRADIENT STUFF
        # self.fd_step = xp.cbrt(xp.finfo(float).eps)
        self.fd_step = 1e-3
       
        self.noise_stddev = 7e-8 # To avoid singularity during QR
       
        self.G = xp.identity(self.model.n_hyperparameters)
        self.c_G = xp.identity(self.model.n_hyperparameters)
        self.prev_theta = xp.zeros(self.model.n_hyperparameters)
        self.curr_theta = xp.zeros(self.model.n_hyperparameters)
        self.count = 0
        self.rng = np.random.default_rng()

    def get_evaluation_directions(self):
        ...
        # theta_mat is called gradient_direction_for_finite_difference in INLA
        # Could be gradient_directions_matrix

    def compute_gradient(self):
        ...



    def _transformed_fun(self, phi):
        return self.curr_theta + self.G @ phi

    def _scale(self, x):
        mean = xp.mean(x)
        std = xp.std(x, ddof=1)
        if std < 1e-12:
            return x - mean
        return (x - mean) / std

    def _update_G(self, current_theta):
        self.curr_theta = current_theta
        self.c_G = xp.roll(self.c_G, 1, axis=1)
        xdiff = current_theta - self.prev_theta
        xdiff += get_device(self.rng.normal(0.0, self.noise_stddev, self.model.n_hyperparameters))
        self.c_G[:, 0] = self._scale(xdiff)
        self.G = self.c_G

    def _orthogonalize_G(self):
        try:
            Q, R = xp.linalg.qr(self.G)
            self.G = Q
        except xp.linalg.LinAlgError:
            print("Warning: QR decomposition failed. Resetting G to identity.")
            self.G = xp.identity(self.model.n_hyperparameters)

    def _get_original_grad(self, transformed_grad):
        return xp.linalg.solve(self.G.T, transformed_grad)

    def _update_gradient_basis(self) -> None:
        for i in range(self.model.n_hyperparameters):
            self.gradient_basis[:, self.model.n_hyperparameters - i - 1] = self.gradient_basis[:, self.model.n_hyperparameters - i - 2]
        
    
    def _construct_f_evaluation_points(self, theta_i: NDArray) -> None:
        if self.count > 0:
            self._update_G(theta_i)
        else:
            self.curr_theta = theta_i  # Set the starting point

        self._orthogonalize_G()

        self.prev_theta = xp.copy(theta_i)
        self.count += 1

        # Initialize central difference scheme matrix
        self.eps_mat[:] = self.fd_step * self.gradient_basis
        self.theta_mat[:] = xp.zeros(
            (self.model.theta.size, self.n_f_evaluations), dtype=xp.float64
        )

        self.curr_theta.T

        self.theta_mat[:, 0] = xp.asarray(self.curr_theta.T)

        self.theta_mat[:, 1 : 1 + self.model.n_hyperparameters] += self.eps_mat
        self.theta_mat[
            :, self.model.n_hyperparameters + 1 : self.n_f_evaluations
        ] -= self.eps_mat

        for i in range(1, self.n_f_evaluations):
            self.theta_mat[:, i] = self._transformed_fun(phi=self.theta_mat[:, i]).T

    def _compute_gradient(self):
        for i in range(self.model.n_hyperparameters):
            self.gradient_f[i] = (
                self.f_values_i[i + 1]
                - self.f_values_i[self.model.n_hyperparameters + i + 1]
            ) / (2 * self.fd_step)
