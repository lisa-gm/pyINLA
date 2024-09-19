# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
import math
from numpy.typing import ArrayLike

# from scipy.optimize import minimize
from scipy.sparse import load_npz

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.likelihoods.binomial import BinomialLikelihood
from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.likelihoods.poisson import PoissonLikelihood
from pyinla.models.regression import Regression
from pyinla.models.spatio_temporal import SpatioTemporal
from pyinla.prior_hyperparameters.gaussian import GaussianPriorHyperparameters
from pyinla.prior_hyperparameters.penalized_complexity import (
    PenalizedComplexityPriorHyperparameters,
)
from pyinla.solvers.scipy_solver import ScipySolver
from pyinla.solvers.serinv_solver import SerinvSolver
from pyinla.utils.theta_utils import theta_array2dict, theta_dict2array


class INLA:
    """Integrated Nested Laplace Approximation (INLA).

    Parameters
    ----------
    pyinla_config : Path
        pyinla configuration file.
    """

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        self.pyinla_config = pyinla_config

        self.eps_inner_iteration = self.pyinla_config.eps_inner_iteration

        # --- Load observation vector
        self.y = np.load(pyinla_config.input_dir / "y.npy")
        self.n_observations = self.y.shape[0]

        # --- Load design matrix
        self.a = load_npz(pyinla_config.input_dir / "a.npz")
        self.n_latent_parameters = self.a.shape[1]

        # --- Load latent parameters vector
        try:
            self.x = np.load(pyinla_config.input_dir / "x.npy")
        except FileNotFoundError:
            self.x = np.zeros((self.a.shape[1]), dtype=self.y.dtype)

        self._check_dimensions()

        # --- Initialize model
        if self.pyinla_config.model.type == "regression":
            self.model = Regression(pyinla_config, self.n_latent_parameters)
        elif self.pyinla_config.model.type == "spatio-temporal":
            self.model = SpatioTemporal(pyinla_config, self.n_latent_parameters)
        else:
            raise ValueError(
                f"Model '{self.pyinla_config.model.type}' not implemented."
            )

        # --- Initialize prior hyperparameters
        if self.pyinla_config.prior_hyperparameters.type == "gaussian":
            self.prior_hyperparameters = GaussianPriorHyperparameters(pyinla_config)
        elif self.pyinla_config.prior_hyperparameters.type == "penalized_complexity":
            self.prior_hyperparameters = PenalizedComplexityPriorHyperparameters(
                pyinla_config
            )
        else:
            raise ValueError(
                f"Prior hyperparameters '{self.pyinla_config.prior_hyperparameters.type}' not implemented."
            )

        # --- Initialize likelihood
        if self.pyinla_config.likelihood.type == "gaussian":
            self.likelihood = GaussianLikelihood(pyinla_config, self.n_observations)
        elif self.pyinla_config.likelihood.type == "poisson":
            self.likelihood = PoissonLikelihood(pyinla_config, self.n_observations)
        elif self.pyinla_config.likelihood.type == "binomial":
            self.likelihood = BinomialLikelihood(pyinla_config, self.n_observations)
        else:
            raise ValueError(
                f"Likelihood '{self.pyinla_config.likelihood.type}' not implemented."
            )

        # --- Initialize solver
        if self.pyinla_config.solver.type == "scipy":
            self.solver_Q_prior = ScipySolver(pyinla_config)
            self.solver_Q_conditional = ScipySolver(pyinla_config)
        elif self.pyinla_config.solver.type == "serinv":
            self.solver_Q_prior = SerinvSolver(pyinla_config)
            self.solver_Q_conditional = SerinvSolver(pyinla_config)

        else:
            raise ValueError(
                f"Solver '{self.pyinla_config.solver.type}' not implemented."
            )

        # --- Initialize theta
        self.theta_initial: ArrayLike = theta_dict2array(
            self.model.get_theta_initial(), self.likelihood.get_theta_initial()
        )

    def run(self) -> np.ndarray:
        """Fit the model using INLA."""

        f_init = self._evaluate_f(self.theta_initial)

        return f_init

        # result = minimize(
        #     self.theta_initial,
        #     self._evaluate_f,
        #     self._evaluate_grad_f,
        #     method="BFGS",
        # )

        # if result.success:
        #     print(
        #         "Optimization converged successfully after", result.nit, "iterations."
        #     )
        #     self.theta_star = result.x
        #     return True
        # else:
        #     print("Optimization did not converge.")
        #     return False

    def get_theta_star(self) -> dict:
        """Get the optimal theta."""
        return theta_array2dict(
            self.theta_star,
            self.model.get_theta_initial(),
            self.likelihood.get_theta_initial(),
        )

    def _check_dimensions(self) -> None:
        """Check the dimensions of the model."""
        assert self.y.shape[0] == self.a.shape[0], "Dimensions of y and A do not match."
        assert self.x.shape[0] == self.a.shape[1], "Dimensions of x and A do not match."

    def _evaluate_f(self, theta_i: np.ndarray) -> float:
        theta_model, theta_likelihood = theta_array2dict(
            theta_i, self.model.get_theta_initial(), self.likelihood.get_theta_initial()
        )

        # --- Evaluate the log prior of the hyperparameters
        log_prior = self.prior_hyperparameters.evaluate_log_prior(
            theta_model, theta_likelihood
        )

        # --- Construct the prior precision matrix of the latent parameters
        Q_prior = self.model.construct_Q_prior(theta_model)

        # --- Optimize x (latent parameters) and construct conditional precision matrix
        Q_conditional, self.x_star = self._inner_iteration(Q_prior, self.x, theta_model)

        # --- Evaluate likelihood at the optimized latent parameters x_star
        likelihood = self.likelihood.evaluate_likelihood(
            self.y, self.a, self.x_star, theta_likelihood
        )

        # --- Evaluate the prior of the latent parameters at x_star
        prior_latent_parameters = self._evaluate_prior_latent_parameters(
            Q_prior, self.x_star
        )

        # --- Evaluate the conditional of the latent parameters at x_star
        conditional_latent_parameters = self._evaluate_conditional_latent_parameters(
            Q_conditional, self.x_star
        )

        return (
            log_prior
            + likelihood
            + prior_latent_parameters
            - conditional_latent_parameters
        )

    def _evaluate_grad_f(self):
        pass

    def _inner_iteration(self, Q_prior, x_i, theta_model):
        x_update = np.zeros_like(x_i)
        x_i_norm = 1
        while x_i_norm >= self.eps_inner_iteration:
            x_i[:] += x_update[:]

            Ax = self.a @ x_i

            gradient_likelihood = self.likelihood.evaluate_grad_likelihood(
                Ax,
                self.y,
            )

            rhs = -1 * Q_prior @ x_i + Ax.T @ gradient_likelihood

            hessian_likelihood = self.likelihood.evaluate_hess_likelihood(Ax, self.y)

            Q_conditional = self.model.construct_Q_conditional(
                Q_prior,
                self.a,
                hessian_likelihood,
            )

            self.solver_Q_conditional.cholesky(Q_conditional)

            x_update[:] = self.solver_Q_conditional.solve(rhs)

            x_i_norm = np.norm(x_update)

        return Q_conditional, x_i

    #
    def _evaluate_prior_latent_parameters(self, x_star, Q_prior):
        """Evaluation of the prior of the latent parameters at x_star using the prior precision matrix Q_prior and assuming mean zero.
        The prior of the latent parameters is by definition a multivariate normal distribution with mean 0 and precision matrix Q_prior
        which is evaluated at x_star in log-scale.
        The evaluation requires the computation of the log determinant of Q_prior.
        log normal: 0.5*log(1/(2*pi)^n * |Q_prior|)) - 0.5 * x_star.T @ Q_prior @ x_star
        """

        n = x_star.shape[0]

        self.solver_Q_prior.cholesky(Q_prior)
        logdet_Q_prior = self.solver_Q_prior.logdet()

        log_prior_latent_parameters = (
            -n / 2 * np.log(2 * math.pi)
            + 0.5 * logdet_Q_prior
            - 0.5 * x_star.T @ Q_prior @ x_star
        )

        return log_prior_latent_parameters

    def _evaluate_conditional_latent_parameters(self, x_star, Q_conditional, x_mean):
        """Evaluation of the conditional of the latent parameters at x_star using the conditional precision matrix Q_conditional and the mean x_mean.
        The conditional of the latent parameters is by definition a multivariate normal distribution with mean x_mean and precision matrix Q_conditional
        which is evaluated at x_star in log-scale.
        The evaluation requires the computation of the log determinant of Q_conditional.
        log normal: 0.5*log(1/(2*pi)^n * |Q_conditional|)) - 0.5 * (x_star - x_mean).T @ Q_conditional @ (x_star - x_mean)
        """

        n = x_star.shape[0]

        # get current theta, check if this theta matches the theta used to construct Q_conditional
        # if yes, check if L already computed, if yes -> takes this L
        self.solver_Q_conditional.cholesky(Q_conditional)
        logdet_Q_conditional = self.solver_Q_conditional.logdet()

        log_conditional_latent_parameters = (
            -n / 2 * np.log(2 * math.pi)
            + 0.5 * logdet_Q_conditional
            - 0.5 * (x_star - x_mean).T @ Q_conditional @ (x_star - x_mean)
        )

        return log_conditional_latent_parameters

    def _evaluate_GMRF(self, x_star, solver_instance, Q, x_mean=None):

        # n = x_star.shape[0]

        # write solver function that checks if current theta matches theta of solver
        # self.solver_instance.cholesky(Q)
        pass
