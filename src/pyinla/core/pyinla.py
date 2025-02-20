# Copyright 2024-2025 pyINLA authors. All rights reserved.

import logging
import math

from scipy import optimize

from pyinla import ArrayLike, NDArray, comm_rank, comm_size, xp
from pyinla.configs.pyinla_config import PyinlaConfig
from pyinla.core.model import Model
from pyinla.solvers import DenseSolver, SerinvSolver, SparseSolver
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from pyinla.utils import allreduce, get_device, get_host, print_msg, set_device

xp.set_printoptions(precision=8, suppress=True, linewidth=150)


class PyINLA:
    """PyINLA is a Python implementation of the Integrated Nested
    Laplace Approximation (INLA) method.
    """

    def __init__(
        self,
        model: Model,
        config: PyinlaConfig,
    ) -> None:
        """Initializes the PyINLA object.

        Parameters
        ----------
        model : Model
            Model object from the pyINLA Models library.
        config : PyinlaConfig
            Configuration object for the PyINLA solver.

        Returns
        -------
        None
        """
        # --- Initialize model
        self.model = model

        # --- Initialize PyINLA
        self.config = config

        self.inner_iteration_max_iter = self.config.inner_iteration_max_iter
        self.eps_inner_iteration = self.config.eps_inner_iteration
        self.eps_gradient_f = self.config.eps_gradient_f

        # --- Configure HPC
        set_device(comm_rank, comm_size)

        self.n_f_evaluations = 2 * self.model.n_hyperparameters + 1

        # --- Initialize solver
        if self.config.solver.type == "dense":
            self.solver = DenseSolver(
                config=self.config.solver,
                n=self.model.n_latent_parameters,
            )
        elif self.config.solver.type == "scipy":
            self.solver = SparseSolver(
                config=self.config.solver,
            )
        elif self.config.solver.type == "serinv":
            diagonal_blocksize: int = 0
            arrowhead_blocksize: int = 0
            n_diag_blocks: int = 0

            # Check the model compute parameters
            if isinstance(self.model.submodels[0], SpatioTemporalSubModel):
                diagonal_blocksize = self.model.submodels[0].ns
                n_diag_blocks = self.model.submodels[0].nt
            else:
                logging.critical("Trying to instanciate Serinv solver on non-ST model.")
                raise ValueError(
                    "Serinv solver is not made for non spatio-temporal models."
                )

            for i in range(1, len(self.model.submodels)):
                if isinstance(self.model.submodels[i], RegressionSubModel):
                    arrowhead_blocksize += self.model.submodels[i].n_latent_parameters
                else:
                    logging.critical(
                        "While measuring the number of arrowhead elements, bumping into a non-supported submodel."
                    )
                    raise ValueError(
                        "Only regression submodels are currently supported in the arrowhead shape of the Serinv solver."
                    )

            self.solver = SerinvSolver(
                config=self.config.solver,
                diagonal_blocksize=diagonal_blocksize,
                arrowhead_blocksize=arrowhead_blocksize,
                n_diag_blocks=n_diag_blocks,
            )

        # --- Set up recurrent variables
        self.gradient_f = xp.zeros(self.model.n_hyperparameters, dtype=xp.float64)
        self.f_values_i = xp.zeros(self.n_f_evaluations, dtype=xp.float64)
        self.eta = xp.zeros_like(self.model.y, dtype=xp.float64)
        self.x_update = xp.zeros_like(self.model.x, dtype=xp.float64)
        self.eps_mat = xp.zeros(
            (self.model.n_hyperparameters, self.model.n_hyperparameters),
            dtype=xp.float64,
        )
        self.theta_mat = xp.zeros(
            (self.model.theta.size, self.n_f_evaluations), dtype=xp.float64
        )

        # --- Metrics
        self.f_values: ArrayLike = []
        self.theta_values: ArrayLike = []

        print_msg("Initial theta:", self.model.theta)

        logging.info("PyINLA initialized.")
        print_msg("PyINLA initialized.", flush=True)

    def run(self) -> optimize.OptimizeResult:
        """Fit the model using INLA.

        Parameters
        ----------
        None

        Returns
        -------
        minimization_result : scipy.optimize.OptimizeResult
            Result of the optimization procedure.
        """

        if len(self.model.theta) == 0:
            # Only run the inner iteration
            print_msg("No hyperparameters, just running inner iteration.")
            self.f_value = self._evaluate_f(self.model.theta)

            minimization_result: dict = {
                "theta": self.model.theta,
                "x": self.model.x,
                "f": self.f_value,
            }
        else:
            print_msg("Starting optimization.")
            self.iter = 0

            # Start the minimization procedure
            def callback(intermediate_result: optimize.OptimizeResult):
                theta_i = intermediate_result.x.copy()
                fun_i = intermediate_result.fun
                self.iter += 1

                # Format the output
                theta_str = ", ".join(f"{theta: .6f}" for theta in theta_i)
                gradient_str = ", ".join(
                    f"{grad: .6f}" for grad in get_host(self.gradient_f)
                )

                print_msg(
                    f"Iteration: {self.iter:2d} | "
                    f"Theta: [{theta_str}] | "
                    f"Function Value: {fun_i: .6f} | "
                    f"Gradient: [{gradient_str}]",
                    flush=True,
                )

                self.theta_values.append(theta_i)
                self.f_values.append(fun_i)

            scipy_result = optimize.minimize(
                fun=self._objective_function,
                x0=get_host(self.model.theta),
                method="L-BFGS-B",
                jac=self.config.minimize.jac,
                options={
                    "maxiter": self.config.minimize.max_iter,
                    "gtol": self.config.minimize.gtol,
                    # "c1": self.config.minimize.c1,
                    # "c2": self.config.minimize.c2,
                    "disp": self.config.minimize.disp,
                    "ftol": 1e-18,
                },
                callback=callback,
            )

            # MEMO:
            # From here rank 0 own the optimized theta_star and the
            # corresponding x_star. Other ranks own garbage thetas
            # ... Could be bcast or other things
            if scipy_result.success:
                print_msg(
                    "Optimization converged successfully after",
                    self.iter,
                    "iterations.",
                    flush=True,
                )
            else:
                print_msg("Optimization did not converge.", flush=True)

            minimization_result: dict = {
                "theta": scipy_result.x,
                "x": self.model.x,
                "f": scipy_result.fun,
                "grad_f": self.gradient_f,
                "f_values": self.f_values,
                "theta_values": self.theta_values,
            }

        return minimization_result

    def _objective_function(
        self,
        theta_i: NDArray,
    ) -> tuple:
        """Objective function to optimize.

        Parameters
        ----------
        theta_i : NDArray
            Hyperparameters theta.

        Returns
        -------
        objective_function_evalutation : tuple
            Function value f(theta) evaluated at theta_i and its gradient.
        """
        # Generate theta matrix with different theta's to evaluate
        # currently central difference scheme is used for gradient
        self.f_values_i[:] = 0.0

        # task to rank assignment
        task_to_rank = xp.zeros(self.n_f_evaluations, dtype=int)

        for i in range(self.n_f_evaluations):
            task_to_rank[i] = i % comm_size

        # Initialize central difference scheme matrix
        self.eps_mat[:] = self.eps_gradient_f * xp.eye(self.model.n_hyperparameters)
        self.theta_mat[:] = xp.repeat(
            get_device(theta_i).reshape(-1, 1), self.n_f_evaluations, axis=1
        )
        self.theta_mat[:, 1 : 1 + self.model.n_hyperparameters] += self.eps_mat
        self.theta_mat[
            :, self.model.n_hyperparameters + 1 : self.n_f_evaluations
        ] -= self.eps_mat

        for i in range(self.n_f_evaluations - 1, -1, -1):
            if task_to_rank[i] == comm_rank:
                self.f_values_i[i] = self._evaluate_f(theta_i=self.theta_mat[:, i])

        allreduce(self.f_values_i, op="sum")

        # Compute gradient using central difference scheme
        for i in range(self.model.n_hyperparameters):
            self.gradient_f[i] = (
                self.f_values_i[i + 1]
                - self.f_values_i[self.model.n_hyperparameters + i + 1]
            ) / (2 * self.eps_gradient_f)

        return (get_host(self.f_values_i[0]), get_host(self.gradient_f))

    def _evaluate_f(
        self,
        theta_i: NDArray,
    ) -> float:
        """Evaluate the objective function f(theta) = log(p(theta|y)).

        Parameters
        ----------
        theta_i : NDArray
            Hyperparameters theta.

        Returns
        -------
        objective_function_evalutation : float
            Function value f(theta) evaluated at theta_i.

        Notes
        -----
        The objective function f(theta) is an approximation of the
        log posterior of the hyperparameters theta evaluated at theta_i
        in log-scale. Consisting of the following 4 terms: log prior
        hyperparameters, log likelihood, log prior of the latent parameters,
        and log conditional of the latent parameters.
        """
        self.model.theta[:] = theta_i

        # --- Evaluate the log prior of the hyperparameters
        log_prior_hyperparameters: float = (
            self.model.evaluate_log_prior_hyperparameters()
        )
        print("log_prior_hyperparameters: ", log_prior_hyperparameters)

        # --- Construct the prior precision matrix of the latent parameters
        self.model.construct_Q_prior()

        # --- Optimize x (latent parameters) and construct conditional precision matrix
        logdet_Q_conditional: float = self._inner_iteration()

        # --- Evaluate likelihood at the optimized latent parameters x_star
        likelihood: float = self.model.likelihood.evaluate_likelihood(
            eta=self.eta,
            y=self.model.y,
            theta=self.model.theta[self.model.hyperparameters_idx[-1] :],
        )
        print("likelihood: ", likelihood)

        # --- Evaluate the prior of the latent parameters at x_star
        prior_latent_parameters: float = self._evaluate_prior_latent_parameters(
            x_star=self.model.x
        )
        print("prior_latent_parameters: ", prior_latent_parameters)

        # --- Evaluate the conditional of the latent parameters at x_star
        conditional_latent_parameters = self._evaluate_conditional_latent_parameters(
            logdet_Q_conditional
        )
        print("conditional_latent_parameters: ", conditional_latent_parameters)

        f_theta: float = -1.0 * (
            log_prior_hyperparameters
            + likelihood
            + prior_latent_parameters
            - conditional_latent_parameters
        )

        return f_theta

    def _inner_iteration(
        self,
    ) -> float:
        """Inner iteration to optimize the latent parameters x.

        Parameters
        ----------
        None

        Returns
        -------
        logdet : float
            Log determinant of the conditional precision matrix Q_conditional.
        """
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
            self.solver.cholesky(A=Q_conditional)

            rhs: NDArray = self.model.construct_information_vector(
                self.eta, self.model.x
            )
            self.x_update[:] = self.solver.solve(
                rhs=rhs,
            )

            x_i_norm = xp.linalg.norm(self.x_update)
            counter += 1

        logdet: float = self.solver.logdet()

        return logdet

    def _evaluate_prior_latent_parameters(
        self,
        x_star: NDArray,
    ) -> float:
        """Evaluation of the prior of the latent parameters at x using
        the prior precision matrix Q_prior and assuming mean zero.

        Parameters
        ----------
        x : NDArray
            Latent parameters.

        Returns
        -------
        logprior : float
            Log prior of the latent parameters evaluated at x

        Notes
        -----
        The prior of the latent parameters is by definition a multivariate normal
        distribution with mean 0 and precision matrix Q_prior which is evaluated at
        x in log-scale. The evaluation requires the computation of the log
        determinant of Q_prior.
        Log normal:
        .. math:: 0.5*log(1/(2*\pi)^n * |Q_prior|)) - 0.5 * x.T Q_prior x
        """
        n: int = x_star.shape[0]

        self.solver.cholesky(self.model.Q_prior)

        # with time_range('getLogDet', color_id=0):
        logdet_Q_prior: float = self.solver.logdet()

        # with time_range('compute_xtQx', color_id=0):
        log_prior_latent_parameters: float = (
            -n / 2 * xp.log(2 * math.pi)
            + 0.5 * logdet_Q_prior
            - 0.5 * x_star.T @ self.model.Q_prior @ x_star
        )

        return log_prior_latent_parameters

    def _evaluate_conditional_latent_parameters(
        self,
        logdet_Q_conditional: float,
    ) -> float:
        """Evaluation of the conditional of the latent parameters at x using
        the conditional precision matrix Q_conditional and the mean x_mean.

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

        Notes
        -----
        The conditional of the latent parameters is by definition a multivariate normal distribution with mean
        x_mean and precision matrix Q_conditional which is evaluated at x in log-scale.
        The evaluation requires the computation of the log determinant of Q_conditional.
        log normal: 0.5*log(1/(2*pi)^n * |Q_conditional|)) - 0.5 * (x - x_mean).T @ Q_conditional @ (x - x_mean)

        TODO: add note for the general way of doing it
        """

        log_conditional_latent_parameters = 0.5 * logdet_Q_conditional

        return log_conditional_latent_parameters
