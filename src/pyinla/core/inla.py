# Copyright 2024 pyINLA authors. All rights reserved.

import math
import os

import numpy as np
import time
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.sparse import diags, load_npz, sparray

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.likelihoods.binomial import BinomialLikelihood
from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.likelihoods.poisson import PoissonLikelihood
from pyinla.models.regression import RegressionModel
from pyinla.models.spatio_temporal import SpatioTemporalModel
from pyinla.prior_hyperparameters.gaussian import GaussianPriorHyperparameters
from pyinla.prior_hyperparameters.penalized_complexity import (
    PenalizedComplexityPriorHyperparameters,
)
from pyinla.solvers.scipy_solver import ScipySolver
from pyinla.solvers.serinv_solver import SerinvSolver
from pyinla.utils.finite_difference_stencils import (
    gradient_finite_difference_5pt,
    hessian_diag_finite_difference_5pt,
)
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

        self.minimize_max_iter = self.pyinla_config.minimize_max_iter
        self.inner_iteration_max_iter = self.pyinla_config.inner_iteration_max_iter
        self.eps_inner_iteration = self.pyinla_config.eps_inner_iteration
        self.eps_gradient_f = self.pyinla_config.eps_gradient_f

        # # --- Load observation vector
        self.y = np.load(pyinla_config.input_dir / "y.npy")
        self.n_observations = self.y.shape[0]

        # --- Load design matrix
        self.a = load_npz(pyinla_config.input_dir / "a.npz")
        self.n_latent_parameters = self.a.shape[1]

        # --- Load latent parameters vector
        try:
            self.x = np.load(pyinla_config.input_dir / "x.npy")
        except FileNotFoundError:
            self.x = np.ones((self.a.shape[1]), dtype=float)

        print("x[:10] : ", self.x[:10])

        self._check_dimensions()

        # --- Initialize model
        if self.pyinla_config.model.type == "regression":
            self.model = RegressionModel(pyinla_config, self.n_latent_parameters)
            print("Regression model initialized.")
        elif self.pyinla_config.model.type == "spatio-temporal":
            self.model = SpatioTemporalModel(pyinla_config, self.n_latent_parameters)
            print("Spatio-temporal model initialized.")
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
            print("Gaussian likelihood initialized.")
        elif self.pyinla_config.likelihood.type == "poisson":
            self.likelihood = PoissonLikelihood(pyinla_config, self.n_observations)
            print("Poisson likelihood initialized.")
        elif self.pyinla_config.likelihood.type == "binomial":
            self.likelihood = BinomialLikelihood(pyinla_config, self.n_observations)
            print("Binomial likelihood initialized.")
        else:
            raise ValueError(
                f"Likelihood '{self.pyinla_config.likelihood.type}' not implemented."
            )

        # --- Initialize solver
        num_threads = os.getenv("OMP_NUM_THREADS")
        print(f"OMP_NUM_THREADS: {num_threads}")

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
            self.model.get_theta(), self.likelihood.get_theta()
        )

        self.theta = self.theta_initial
        self.f_value = 1e10

        self.counter = 0
        self.min_f = 1e10

        # --- set up recurrent variables
        self.Q_conditional: sparray = None

        print("INLA initialized.", flush=True)
        print("Model:", self.pyinla_config.model.type, flush=True)
        if len(self.model.get_theta()) > 0:
            print(
                "Prior hyperparameters model:",
                self.pyinla_config.prior_hyperparameters.type,
                flush=True,
            )
            print(
                f"  Prior theta model - spatial range. mean : {self.prior_hyperparameters.mean_theta_spatial_range}, precision : {self.prior_hyperparameters.precision_theta_spatial_range}\n"
                f"  Prior theta model - temporal range. mean : {self.prior_hyperparameters.mean_theta_temporal_range}, precision : {self.prior_hyperparameters.precision_theta_temporal_range}\n"
                f"  Prior theta model - spatio-temporal variation. mean : {self.prior_hyperparameters.mean_theta_spatio_temporal_variation}, precision : {self.prior_hyperparameters.precision_theta_spatio_temporal_variation}",
                flush=True,
            )

        if len(self.likelihood.get_theta()) > 0:
            print(
                "   Prior hyperparameters likelihood:",
                self.pyinla_config.prior_hyperparameters.type,
                flush=True,
            )
            print(
                f"  Prior theta likelihood. Mean : {self.prior_hyperparameters.mean_theta_observations}, precision : {self.prior_hyperparameters.precision_theta_observations}\n",
                flush=True,
            )
            print("   Initial theta:", self.theta_initial, flush=True)
        print("   Likelihood:", self.pyinla_config.likelihood.type, flush=True)

    def run(self) -> np.ndarray:
        """Fit the model using INLA."""

        self.f_value = self._evaluate_f(self.theta_initial)
        print(f"Initial function value: {self.f_value}", flush=True)

        if len(self.theta) > 0:
            grad_f_init = self._evaluate_gradient_f(self.theta_initial)
            print(f"Initial gradient: {grad_f_init}", flush=True)

            result = minimize(
                self._evaluate_f,
                self.theta_initial,
                method="BFGS",
                jac=self._evaluate_gradient_f,
                options={
                    "maxiter": self.minimize_max_iter,
                    "gtol": 1e-1,
                    "disp": True,
                },
            )

            if result.success:
                print(
                    "Optimization converged successfully after",
                    result.nit,
                    "iterations.",
                    flush=True,
                )
                self.theta = result.x
                theta_model, theta_likelihood = theta_array2dict(
                    self.theta, self.model.get_theta(), self.likelihood.get_theta()
                )
                # print("Optimal theta:", self.theta_star)
                # print("Latent parameters:", self.x)
            else:
                print("Optimization did not converge.", flush=True)
                return False

            print("counter:", self.counter, flush=True)

        # TODO: check that Q_conditional was constructed using the right theta
        if self.Q_conditional is not None:
            self._placeholder_marginals_latent_parameters(self.Q_conditional)
        else:
            print("Q_conditional not defined.", flush=True)
            raise ValueError

        return True

    def get_theta_star(self) -> dict:
        """Get the optimal theta."""
        return theta_array2dict(
            self.theta,
            self.model.get_theta(),
            self.likelihood.get_theta(),
        )

    def _check_dimensions(self) -> None:
        """Check the dimensions of the model."""
        assert self.y.shape[0] == self.a.shape[0], "Dimensions of y and A do not match."
        assert self.x.shape[0] == self.a.shape[1], "Dimensions of x and A do not match."

    def _evaluate_f(self, theta_i: np.ndarray) -> float:
        """evaluate the objective function f(theta) = log(p(theta|y)).

        Notes
        -----

        The objective function f(theta) is an approximation of the log posterior of the hyperparameters theta evaluated at theta_i in log-scale.
        Consisting of the following 4 terms: log prior hyperparameters, log likelihood, log prior of the latent parameters, and log conditional of the latent parameters.

        Parameters
        ----------
        theta_i : ArrayLike
            Hyperparameters theta.

        Returns
        -------
        objective_function_evalutation : float
            Function value f(theta) evaluated at theta_i.
        """

        print("Evaluate f()", flush=True)

        self.theta = theta_i

        theta_model, theta_likelihood = theta_array2dict(
            theta_i, self.model.get_theta(), self.likelihood.get_theta()
        )

        # --- Evaluate the log prior of the hyperparameters
        tic = time.perf_counter()
        log_prior_hyperparameters = self.prior_hyperparameters.evaluate_log_prior(
            theta_model, theta_likelihood
        )
        toc = time.perf_counter()
        print("   (1/6) evaluate_log_prior time:", toc - tic, flush=True)

        # --- Construct the prior precision matrix of the latent parameters
        tic = time.perf_counter()
        Q_prior = self.model.construct_Q_prior(theta_model)
        toc = time.perf_counter()
        print("   (2/6) construct_Q_prior time:", toc - tic, flush=True)

        # --- Optimize x (latent parameters) and construct conditional precision matrix
        tic = time.perf_counter()
        self.Q_conditional, self.x = self._inner_iteration(
            Q_prior, self.x, theta_likelihood
        )
        toc = time.perf_counter()
        print("   (3/6) inner_iteration time:", toc - tic, flush=True)

        # --- Evaluate likelihood at the optimized latent parameters x_star
        tic = time.perf_counter()
        eta = self.a @ self.x
        likelihood = self.likelihood.evaluate_likelihood(eta, self.y, theta_likelihood)
        toc = time.perf_counter()
        print("   (4/6) evaluate_likelihood time:", toc - tic, flush=True)

        # --- Evaluate the prior of the latent parameters at x_star
        tic = time.perf_counter()
        prior_latent_parameters = self._evaluate_prior_latent_parameters(
            Q_prior, self.x
        )
        toc = time.perf_counter()
        print("   (5/6) evaluate_prior_latent_parameters time:", toc - tic, flush=True)

        # --- Evaluate the conditional of the latent parameters at x_star
        tic = time.perf_counter()
        conditional_latent_parameters = self._evaluate_conditional_latent_parameters(
            self.Q_conditional, self.x, self.x
        )
        toc = time.perf_counter()
        print(
            "   (6/6) evaluate_conditional_latent_parameters time:",
            toc - tic,
            flush=True,
        )

        f_theta = -1 * (
            log_prior_hyperparameters
            + likelihood
            + prior_latent_parameters
            - conditional_latent_parameters
        )

        # print(f"theta: {theta_i},      Function value: {f_theta}")

        if f_theta < self.min_f:
            self.min_f = f_theta
            self.counter += 1
            print(f"theta: {theta_i},      Function value: {f_theta}", flush=True)
            # print(f"Minimum function value: {self.min_f}. Counter: {self.counter}")

        return f_theta

    def _evaluate_gradient_f(self, theta_i: np.ndarray) -> np.ndarray:
        """evaluate the gradient of the objective function f(theta) = log(p(theta|y)).

        Notes
        -----
        Evaluate the gradient of the objective function f(theta) = log(p(theta|y)) wrt to theta
        using a finite difference approximation. For now implement only central difference scheme.

        Returns
        -------
        grad_f : np.ndarray
            Gradient of the objective function f(theta) evaluated at theta_i.

        """

        print("Evaluate gradient_f()", flush=True)

        tic = time.perf_counter()
        dim_theta = theta_i.shape[0]
        grad_f = np.zeros(dim_theta)

        # TODO: Theses evaluations are independant and can be performed
        # in parallel.
        for i in range(dim_theta):
            theta_plus = theta_i.copy()
            theta_minus = theta_i.copy()

            theta_plus[i] += self.eps_gradient_f
            theta_minus[i] -= self.eps_gradient_f

            f_plus = self._evaluate_f(theta_plus)
            f_minus = self._evaluate_f(theta_minus)

            grad_f[i] = (f_plus - f_minus) / (2 * self.eps_gradient_f)
        toc = time.perf_counter()
        print("   evaluate_gradient_f time:", toc - tic, flush=True)

        print(f"Gradient: {grad_f}", flush=True)

        return grad_f

    def _inner_iteration(
        self, Q_prior: sparray, x_i: ArrayLike, theta_likelihood: dict
    ):
        x_update = np.zeros_like(x_i)
        x_i_norm = 1

        print("   Starting inner iteration", flush=True)

        counter = 0
        while x_i_norm >= self.eps_inner_iteration:
            if counter > self.inner_iteration_max_iter:
                raise ValueError(
                    f"Inner iteration did not converge after {counter} iterations."
                )
            print(f"      inner iteration {counter} norm: {x_i_norm}", flush=True)

            tic = time.perf_counter()
            x_i[:] += x_update[:]
            eta = self.a @ x_i

            # TODO: need to vectorize !!
            gradient_likelihood = gradient_finite_difference_5pt(
                self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
            )
            toc = time.perf_counter()
            print("         evaluate_likelihood time:", toc - tic, flush=True)

            tic = time.perf_counter()
            rhs = -1 * Q_prior @ x_i + self.a.T @ gradient_likelihood

            # TODO: need to vectorize
            hessian_likelihood_diag = hessian_diag_finite_difference_5pt(
                self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
            )
            hessian_likelihood = diags(hessian_likelihood_diag)
            toc = time.perf_counter()
            print(
                "         hessian_diag_finite_difference_5pt time:",
                toc - tic,
                flush=True,
            )

            tic = time.perf_counter()
            Q_conditional = self.model.construct_Q_conditional(
                Q_prior,
                self.a,
                hessian_likelihood,
            )
            toc = time.perf_counter()
            print("         construct_Q_conditional time:", toc - tic, flush=True)

            tic = time.perf_counter()
            self.solver_Q_conditional.cholesky(Q_conditional)
            toc = time.perf_counter()
            print("         cholesky time:", toc - tic, flush=True)

            tic = time.perf_counter()
            x_update[:] = self.solver_Q_conditional.solve(rhs)
            toc = time.perf_counter()
            print("         solve time:", toc - tic, flush=True)

            x_i_norm = np.linalg.norm(x_update)
            counter += 1
            print(f"Inner iteration {counter} norm: {x_i_norm}")

        print("Inner iteration converged after", counter, "iterations.")
        return Q_conditional, x_i

    def _evaluate_prior_latent_parameters(self, Q_prior: sparray, x: ArrayLike):
        """Evaluation of the prior of the latent parameters at x using the prior precision
        matrix Q_prior and assuming mean zero.

        Parameters
        ----------
        Q_prior : ArrayLike
            Prior precision matrix.
        x : ArrayLike
            Latent parameters.

        Notes
        -----
        The prior of the latent parameters is by definition a multivariate normal
        distribution with mean 0 and precision matrix Q_prior which is evaluated at
        x in log-scale. The evaluation requires the computation of the log
        determinant of Q_prior.
        Log normal:
        .. math:: 0.5*log(1/(2*\pi)^n * |Q_prior|)) - 0.5 * x.T Q_prior x

        Returns
        -------
        logprior : float
            Log prior of the latent parameters evaluated at x
        """

        n = x.shape[0]

        self.solver_Q_prior.cholesky(Q_prior)
        logdet_Q_prior = self.solver_Q_prior.logdet()

        log_prior_latent_parameters = (
            -n / 2 * np.log(2 * math.pi)
            + 0.5 * logdet_Q_prior
            - 0.5 * x.T @ Q_prior @ x
        )

        return log_prior_latent_parameters

    def _evaluate_conditional_latent_parameters(self, Q_conditional, x, x_mean):
        """Evaluation of the conditional of the latent parameters at x using the conditional precision matrix Q_conditional and the mean x_mean.

        Notes
        -----
        The conditional of the latent parameters is by definition a multivariate normal distribution with mean x_mean and precision matrix Q_conditional
        which is evaluated at x in log-scale.
        The evaluation requires the computation of the log determinant of Q_conditional.
        log normal: 0.5*log(1/(2*pi)^n * |Q_conditional|)) - 0.5 * (x - x_mean).T @ Q_conditional @ (x - x_mean)

        Parameters
        ----------
        Q_conditional : ArrayLike
            Conditional precision matrix.
        x : ArrayLike
            Latent parameters.
        x_mean : ArrayLike
            Mean of the latent parameters.

        Returns
        -------
        log_conditional : float
            Log conditional of the latent parameters evaluated at x
        """

        # n = x.shape[0]

        # get current theta, check if this theta matches the theta used to construct Q_conditional
        # if yes, check if L already computed, if yes -> takes this L
        self.solver_Q_conditional.cholesky(Q_conditional)
        logdet_Q_conditional = self.solver_Q_conditional.logdet()

        # TODO: evaluate it at the mean -> this becomes zero ...
        # log_conditional_latent_parameters = (
        #     -n / 2 * np.log(2 * math.pi)
        #     + 0.5 * logdet_Q_conditional
        #     - 0.5 * (x - x_mean).T @ Q_conditional @ (x - x_mean)
        # )

        log_conditional_latent_parameters = 0.5 * logdet_Q_conditional

        return log_conditional_latent_parameters

    def _evaluate_GMRF(self, x_star, solver_instance, Q, x_mean=None):
        # n = x_star.shape[0]

        # write solver function that checks if current theta matches theta of solver
        # self.solver_instance.cholesky(Q)
        pass

    def _placeholder_marginals_latent_parameters(self, Q_conditional):
        # self.solver_Q_conditional.cholesky(Q_conditional)
        # TODO: add proper check that current theta & theta of Q_conditional match
        theta_model, _ = theta_array2dict(
            self.theta, self.model.get_theta(), self.likelihood.get_theta()
        )
        if self.model.get_theta() != theta_model:
            print("theta of Q_conditional does not match current theta")
            raise ValueError

        self.solver_Q_conditional.full_inverse()
        Q_inverse_selected = self.solver_Q_conditional.extract_selected_inverse(
            self.solver_Q_conditional.A_inv
        )

        # min_size = min(self.n_latent_parameters, 6)
        # print(f"Q_inverse_selected[:{min_size}, :{min_size}]: \n", Q_inverse_selected[:min_size, :min_size].toarray())

        latent_parameters_marginal_sd = np.sqrt(Q_inverse_selected.diagonal())
        print(
            f"standard deviation fixed effects: {latent_parameters_marginal_sd[-self.pyinla_config.model.n_fixed_effects:]}"
        )

        return Q_inverse_selected
