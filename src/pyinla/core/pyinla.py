# Copyright 2024 pyINLA authors. All rights reserved.

import math
from warnings import warn

from scipy.optimize import minimize

from pyinla import ArrayLike, NDArray, comm_rank, comm_size, xp
from pyinla.configs.pyinla_config import PyinlaConfig
from pyinla.core.model import Model
from pyinla.solvers import DenseSolver, SerinvSolver, SparseSolver
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from pyinla.utils import allreduce, print_msg, set_device


class PyINLA:
    """Integrated Nested Laplace Approximation (INLA).

    Parameters
    ----------
    config : Path
        pyinla configuration file.
    """

    def __init__(
        self,
        model: Model,
        config: PyinlaConfig,
    ) -> None:
        # --- Initialize model
        self.model = model

        # --- Initialize PyINLA
        self.config = config

        self.inner_iteration_max_iter = self.config.inner_iteration_max_iter
        self.eps_inner_iteration = self.config.eps_inner_iteration
        self.eps_gradient_f = self.config.eps_gradient_f

        # --- Configure HPC
        set_device(comm_rank)

        print("END.")
        exit()

        # --- Initialize solver

        # Get the solver parameter for the sparsity pattern
        diagonal_blocksize = 0
        arrowhead_size = 0
        n_diag_blocks = 0

        # Check the first submodel - if ST, get diagonal_blocksize and n_diag_blocks (bt or bta)
        if isinstance(self.model.submodels[0], SpatioTemporalSubModel):
            diagonal_blocksize = self.model.submodels[0].ns
            n_diag_blocks = self.model.submodels[0].nt

        for i in range(1, len(self.model.submodels)):
            if isinstance(self.model.submodels[i], RegressionSubModel):
                arrowhead_size += self.model.submodels[i].n_latent_parameters
            else:
                warn(
                    "Only regression submodels are supported for now in the arrowhead."
                )

        if diagonal_blocksize == 0 or n_diag_blocks == 0:
            if self.config.solver.type == "scipy":
                self.solver = SparseSolver(
                    solver_config=self.config.solver,
                )
            else:
                if self.config.solver.type == "serinv":
                    warn(
                        "SerinvSolver doesn't support non-ST models. Defaulting to DenseSolver."
                    )
                self.solver = DenseSolver(
                    solver_config=self.config.solver,
                    kwargs={"n": self.model.n_latent_parameters},
                )
        else:
            if self.config.solver.type == "dense":
                warn(
                    "DenseSolver is being instanciated to solve ST models (not-optimal)."
                )
                self.solver = DenseSolver(
                    solver_config=self.config.solver,
                    kwargs={"n": self.model.n_latent_parameters},
                )
            elif self.config.solver.type == "scipy":
                self.solver = SparseSolver(
                    solver_config=self.config.solver,
                )
            elif self.config.solver.type == "serinv":
                self.solver = SerinvSolver()

        # ...
        self.n_f_evaluations = 2 * self.model.n_hyperparameters + 1

        # --- Set up recurrent variables
        self.gradient_f = xp.zeros(self.model.n_hyperparameters)
        self.f_values_i = xp.zeros(self.n_f_evaluations)
        self.eta = xp.zeros_like(self.model.y)
        self.x_update = xp.empty_like(self.model.x)

        # --- Metrics
        self.f_values: ArrayLike = []
        self.theta_values: ArrayLike = []

        print_msg("INLA initialized.", flush=True)

    def run(self) -> bool:
        """Fit the model using INLA."""

        if len(self.model.theta) > 0:

            def callback(xk):
                theta_i = xk.x.copy()
                fun_i = self.fun.copy()
                iter = xk.nit

                print_msg(
                    f"iter: {iter}: theta: {theta_i}, f: {fun_i}",
                    flush=True,
                )

                self.theta_values.append(theta_i)
                self.f_values.append(fun_i)

            result = minimize(
                self._objective_function,
                self.model.theta,
                method="BFGS",
                jac=self.config.minimize.jac,
                options={
                    "maxiter": self.config.minimize.max_iter,
                    "gtol": self.config.minimize.gtol,
                    "c1": self.config.minimize.c1,
                    "c2": self.config.minimize.c2,
                    "disp": self.config.minimize.disp,
                },
                callback=callback,
            )

            # MEMO:
            # From here rank 0 own the optimized theta_star and the
            # corresponding x_star. Other ranks own garbage thetas
            # ... Could be bcast or other thing

            if result.success:
                print_msg(
                    "Optimization converged successfully after",
                    result.nit,
                    "iterations.",
                    flush=True,
                )
                return True
            else:
                print_msg("Optimization did not converge.", flush=True)
                return False

        # Only run inner iteration
        else:
            print("in evaluate f.")
            self.f_value = self._evaluate_f(self.model.theta)

        return True

    def _objective_function(self, theta_i: NDArray) -> float:
        """Objective function for optimization."""

        # generate theta matrix with different theta's to evaluate
        # currently central difference scheme is used for gradient
        self.f_values_i[:] = 0.0

        # task to rank assignment
        task_to_rank = xp.zeros(self.n_f_evaluations, dtype=int)

        for i in range(self.n_f_evaluations):
            task_to_rank[i] = i % comm_size

        # Initialize central difference scheme matrix
        # TODO: Pre-allocate epsMat
        epsMat = self.eps_gradient_f * xp.eye(self.model.n_hyperparameters)
        theta_mat = xp.repeat(theta_i.reshape(-1, 1), self.n_f_evaluations, axis=1)
        theta_mat[:, 1 : 1 + self.model.n_hyperparameters] += epsMat
        theta_mat[:, self.model.n_hyperparameters + 1 : self.n_f_evaluations] -= epsMat

        for i in range(self.n_f_evaluations - 1, -1, -1):
            if task_to_rank[i] == comm_rank:
                self.f_values_i[i] = self._evaluate_f(theta_i=theta_mat[:, i])

        allreduce(self.f_values_i, op="sum")

        # Compute gradient using central difference scheme
        for i in range(self.model.n_hyperparameters):
            self.gradient_f[i] = (
                self.f_values_i[i + 1]
                - self.f_values_i[i + self.model.n_hyperparameters + 1]
            ) / (2 * self.eps_gradient_f)

        return (self.f_values_i[0], self.gradient_f)

    def _evaluate_f(self, theta_i: NDArray) -> float:
        """evaluate the objective function f(theta) = log(p(theta|y)).

        Notes
        -----

        The objective function f(theta) is an approximation of the log posterior of the hyperparameters theta evaluated at theta_i in log-scale.
        Consisting of the following 4 terms: log prior hyperparameters, log likelihood, log prior of the latent parameters, and log conditional of the latent parameters.

        Parameters
        ----------
        theta_i : NDArray
            Hyperparameters theta.

        Returns
        -------
        objective_function_evalutation : float
            Function value f(theta) evaluated at theta_i.
        """
        self.model.theta[:] = theta_i

        # --- Evaluate the log prior of the hyperparameters
        log_prior_hyperparameters: float = (
            self.model.evaluate_log_prior_hyperparameters()
        )

        # --- Construct the prior precision matrix of the latent parameters
        self.model.construct_Q_prior()

        # --- Optimize x (latent parameters) and construct conditional precision matrix
        logdet_Q_conditional: float = self._inner_iteration()

        # --- Evaluate likelihood at the optimized latent parameters x_star
        likelihood: float = self.model.likelihood.evaluate_likelihood(
            eta=self.eta,
            y=self.y,
            kwargs={"theta": self.model.theta[self.model.hyperparameters_idx[-1] :]},
        )

        # --- Evaluate the prior of the latent parameters at x_star
        prior_latent_parameters: float = self._evaluate_prior_latent_parameters(
            x_star=self.model.x
        )

        # --- Evaluate the conditional of the latent parameters at x_star
        conditional_latent_parameters = self._evaluate_conditional_latent_parameters(
            logdet_Q_conditional
        )

        f_theta: float = -1.0 * (
            log_prior_hyperparameters
            + likelihood
            + prior_latent_parameters
            - conditional_latent_parameters
        )

        return f_theta

    def _inner_iteration(
        self,
    ):
        self.x_update[:] = 0.0
        x_i_norm: float = 1.0

        counter: int = 0
        while x_i_norm >= self.eps_inner_iteration:
            if counter > self.inner_iteration_max_iter:
                print_msg("current theta value: ", self.model.theta)
                raise ValueError(
                    f"Inner iteration did not converge after {counter} iterations."
                )

            self.model.x[:] += self.x_update[:]
            self.eta[:] = self.model.a @ self.model.x

            Q_conditional = self.model.construct_Q_conditional(self.eta)
            self.solver.cholesky(Q_conditional, sparsity=self.sparsity_Q_conditional)

            rhs: NDArray = self.model.construct_information_vector(
                self.eta, self.model.x
            )
            self.x_update[:] = self.solver.solve(
                rhs, sparsity=self.sparsity_Q_conditional
            )

            x_i_norm = xp.linalg.norm(self.x_update)
            counter += 1

        logdet: float = self.solver.logdet()

        return logdet

    def _evaluate_prior_latent_parameters(self, x_star: NDArray) -> float:
        """Evaluation of the prior of the latent parameters at x using the prior precision
        matrix Q_prior and assuming mean zero.

        Parameters
        ----------
        x : NDArray
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

        n: int = x_star.shape[0]

        self.solver.cholesky(self.model.Q_prior, sparsity=self.sparsity_Q_prior)

        # with time_range('getLogDet', color_id=0):
        logdet_Q_prior: float = self.solver.logdet()

        # with time_range('compute_xtQx', color_id=0):
        log_prior_latent_parameters: float = (
            -n / 2 * xp.log(2 * math.pi)
            + 0.5 * logdet_Q_prior
            - 0.5 * x_star.T @ self.model.Q_prior @ x_star
        )

        return log_prior_latent_parameters

    def _evaluate_conditional_latent_parameters(self, logdet_Q_conditional: float):
        """Evaluation of the conditional of the latent parameters at x using the conditional precision matrix Q_conditional and the mean x_mean.

        Notes
        -----
        The conditional of the latent parameters is by definition a multivariate normal distribution with mean x_mean and precision matrix Q_conditional
        which is evaluated at x in log-scale.
        The evaluation requires the computation of the log determinant of Q_conditional.
        log normal: 0.5*log(1/(2*pi)^n * |Q_conditional|)) - 0.5 * (x - x_mean).T @ Q_conditional @ (x - x_mean)

        TODO: add note for the general way of doing it

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

        log_conditional_latent_parameters = 0.5 * logdet_Q_conditional

        return log_conditional_latent_parameters
