# integration strategy


class INLA:
    """
    ...
    """

    # 1. Class attributes (if any)
    # 2. Initialization
    # 3. Special representation methods
    # 4. Properties (grouped together)
    # 5. Comparison operators (if needed)
    # 6. Arithmetic operators (standard order)
    # 7. Right-hand operators (same order as above)
    # 8. In-place operators (if supported)
    # 9. Other special methods
    # 10. Public methods
    def objective_function(self, theta):
        """Current `dalia._evaluate_f()`

        return f() at given theta

        f of theta is the sum of:
        - log prior of theta
        - log likelihood of data given theta
        - log prior of the latent variables given theta
        - log conditional of the latent parameters

        """
        f_of_theta = (
            self._evaluate_log_prior_theta(theta)
            + self._evaluate_log_likelihood(theta)
            + self._evaluate_log_prior_latent(theta)
            - self._evaluate_log_conditional_latent(theta)
        )

        return f_of_theta

    # 11. Private/protected methods (start with _)
    def _evaluate_log_prior_theta(self, theta):
        """Evaluate the log prior of theta."""
        log_prior = self.model.evaluate_log_prior_hyperparameters()
        return log_prior

    def _evaluate_log_likelihood(self, theta):
        """Evaluate the log likelihood of the data given theta."""
        log_likelihood = self.model.evaluate_likelihood(
            eta=eta,
        )
        return log_likelihood

    def _evaluate_log_prior_latent(self, theta):
        """Evaluate the log prior of the latent variables given theta."""
        self.solver.factorize(self.model.Q_prior, sparsity="bt")
        logdet_Q_prior: float = self.solver.logdet(sparsity="bt")

        log_prior_latent: float = +0.5 * logdet_Q_prior

        if x is not None:
            log_prior_latent -= 0.5 * x.T @ self.model.Q_prior @ x

        return log_prior_latent

    def _evaluate_log_conditional_latent(self, theta):
        """Evaluate the log conditional of the latent parameters."""
        Q_conditional = self.model.construct_Q_conditional(eta)

        self.solver.factorize(A=Q_conditional, sparsity="bta")

        rhs = self.model.construct_information_vector(
            eta,
            x,
        )

        x = self.solver.solve(
            rhs=rhs,
            sparsity="bta",
        )

        # Compute the log determinant of Q_conditional
        logdet_Q_conditional = self.solver.logdet(sparsity="bta")

        if x is None and x_mean is None:
            quadratic_form = 0.0
        # TODO: there is probably a cleaner way to formulate these statements ...
        # the else fails if x_mean is None
        else:
            if x is None and x_mean is not None:
                quadratic_form = x_mean.T @ Q_conditional @ x_mean
            else:
                quadratic_form = (x - x_mean).T @ Q_conditional @ (x - x_mean)

        # Compute the log conditional
        log_conditional_latent = 0.5 * logdet_Q_conditional - 0.5 * quadratic_form

        return log_conditional_latent


# self.model.construct_q_prior(self.model.convert_to_internal_scale(theta))
