# Copyright 2024-2025 pyINLA authors. All rights reserved.

import logging
import math

from scipy import optimize

from pyinla import ArrayLike, NDArray, comm_rank, comm_size, xp, sp
from pyinla.configs.pyinla_config import PyinlaConfig
from pyinla.core.model import Model
from pyinla.solvers import DenseSolver, SerinvSolver, SparseSolver
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from pyinla.utils import allreduce, get_device, get_host, print_msg, set_device, get_active_comm, smartsplit

xp.set_printoptions(precision=8, suppress=True, linewidth=150)

from mpi4py import MPI

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

        # Split the feval communicator
        self.comm_world, self.comm_feval, self.color_feval = smartsplit(
            comm=MPI.COMM_WORLD, 
            n_parallelizable_evaluations=self.n_f_evaluations, 
            tag="feval"
        )
        self.world_size = self.comm_world.Get_size()

        # Split the qeval communicator
        if self.model.is_likelihood_gaussian():
            self.n_qeval = 2
        else:
            self.n_qeval = 1
        _, self.comm_qeval, self.color_qeval = smartsplit(
            comm=self.comm_feval,
            n_parallelizable_evaluations=self.n_qeval,
            tag="qeval"
        )

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
            serinv_parameters = model.get_solver_parameters()
            diagonal_blocksize: int = serinv_parameters["diagonal_blocksize"]
            arrowhead_blocksize: int = serinv_parameters["arrowhead_blocksize"]
            n_diag_blocks: int = serinv_parameters["n_diag_blocks"]

            # Check the model compute parameters
            if diagonal_blocksize is None or n_diag_blocks is None:
                logging.critical(
                    "Trying to instanciate Serinv solver on non-ST model."
                )
                raise ValueError(
                    "Serinv solver is not made for non spatio-temporal models."
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
        # self.eta = xp.zeros_like(self.model.y, dtype=xp.float64)
        # self.x_update = xp.zeros_like(self.model.x, dtype=xp.float64)
        self.eps_mat = xp.zeros(
            (self.model.n_hyperparameters, self.model.n_hyperparameters),
            dtype=xp.float64,
        )
        self.theta_mat = xp.zeros(
            (self.model.theta.size, self.n_f_evaluations), dtype=xp.float64
        )
        self.theta_optimizer = xp.zeros_like(self.model.theta)
        self.theta_optimizer[:] = self.model.theta

        # --- Metrics
        self.f_values: ArrayLike = []
        self.theta_values: ArrayLike = []

        print_msg("Initial theta:", self.model.theta)

        logging.info("PyINLA initialized.")
        print_msg("PyINLA initialized.", flush=True)

        self.i = 0

        self.initial_f_value = 0.

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

                print(
                    f"comm_rank: {comm_rank} | "
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
                x0=get_host(self.theta_optimizer),
                method="L-BFGS-B",
                jac=self.config.minimize.jac,
                options={
                    "maxiter": self.config.minimize.max_iter,
                    "gtol": self.config.minimize.gtol,
                    "ftol": 0.0,
                    "disp": self.config.minimize.disp,
                    "ftol": 1e-18,
                },
                callback=callback,
            )

            # MEMO:
            # From here rank 0 own the optimized theta_star and the
            # corresponding x_star. Other ranks own garbage thetas in 
            # their self.model.theta
            if scipy_result.success:
                print_msg(
                    "Optimization converged successfully after",
                    self.iter,
                    "iterations.\n",
                    "SUCCESS MSG: ",
                    scipy_result.message,
                    flush=True,
                )
            else:
                print_msg(
                    "Optimization did not converge.", 
                    "FAILURE MSG: ",
                    scipy_result.message,
                    flush=True,
                )

            minimization_result: dict = {
                "theta": scipy_result.x,
                "x": get_host(self.model.x),
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

        # Multiprocessing task to rank assignment
        task_mapping = []
        for i in range(self.n_f_evaluations):
                task_mapping.append(i % self.world_size)

        # Initialize central difference scheme matrix
        self.eps_mat[:] = self.eps_gradient_f * xp.eye(self.model.n_hyperparameters)
        self.theta_mat[:] = xp.repeat(
            get_device(theta_i).reshape(-1, 1), self.n_f_evaluations, axis=1
        )
        self.theta_mat[:, 1 : 1 + self.model.n_hyperparameters] += self.eps_mat
        self.theta_mat[
            :, self.model.n_hyperparameters + 1 : self.n_f_evaluations
        ] -= self.eps_mat

        # Proceed to the parallel function evaluation
        for feval_i in range(self.n_f_evaluations - 1, -1, -1):
            # Perform the evaluation in reverse order so that the stored and returned
            # self.x value matches the "bare" hyperparameters evaluation
            if self.color_feval == task_mapping[feval_i]:
                self.f_values_i[feval_i] = self._evaluate_f(
                    theta_i=self.theta_mat[:, feval_i], 
                    comm=self.comm_feval
                )

        # Here carefull on the reduction as it's gonna add the values from all ranks and not only the root of the groups - TODO
        allreduce(self.f_values_i, op="sum", factor= 1/self.comm_feval.Get_size(), comm=self.comm_world)

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
        comm: MPI.Comm,
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

        # --- Construct the prior precision matrix of the latent parameters
        self.model.construct_Q_prior()


        # --- Optimize x and evaluate the conditional of the latent parameters
        if self.model.is_likelihood_gaussian():
            eta = xp.zeros_like(self.model.y, dtype=xp.float64)
            x = xp.zeros_like(self.model.x, dtype=xp.float64)

            # Done by processes "even"
            prior_latent_parameters: float = self._evaluate_prior_latent_parameters()

            # Done by processes "odd"
            Q_conditional = self.model.construct_Q_conditional(eta)
            self.solver.cholesky(A=Q_conditional)
            rhs: NDArray = self.model.construct_information_vector(
                eta, x,
            )
            self.model.x[:] = self.solver.solve(
                rhs=rhs,
            )

            conditional_latent_parameters = self._evaluate_conditional_latent_parameters(
                Q_conditional=Q_conditional,
                x=None,
                x_mean=self.model.x,
            )
        else:
            Q_conditional, self.model.x[:], eta = self._inner_iteration()

            conditional_latent_parameters = self._evaluate_conditional_latent_parameters(
                Q_conditional=Q_conditional,
                x=None,
                x_mean=None,
            )

            prior_latent_parameters: float = self._evaluate_prior_latent_parameters(
                x=self.model.x,
            )

        # --- Evaluate likelihood at the optimized latent parameters x_star
        likelihood: float = self.model.evaluate_likelihood(
            eta=eta,
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
        x_star = self.model.x.copy()
        x_update = xp.zeros_like(self.model.x, dtype=xp.float64)
        x_i_norm: float = 1.0
        eta = xp.zeros_like(self.model.y, dtype=xp.float64)

        counter: int = 0
        while x_i_norm >= self.eps_inner_iteration:
            if counter > self.inner_iteration_max_iter:
                print_msg("Theta value at failing of the inner_iteration: ", self.model.theta, flush=True)
                raise ValueError(
                    f"Inner iteration did not converge after {counter} iterations."
                )

            x_star[:] += x_update
            eta[:] = self.model.a @ x_star

            Q_conditional = self.model.construct_Q_conditional(eta)
            self.solver.cholesky(A=Q_conditional)

            rhs: NDArray = self.model.construct_information_vector(
                eta, x_star,
            )
            x_update[:] = self.solver.solve(
                rhs=rhs,
            )

            x_i_norm = xp.linalg.norm(x_update)
            counter += 1

        return Q_conditional, x_star, eta

    def _evaluate_prior_latent_parameters(
        self,
        x: NDArray = None,
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
        self.solver.cholesky(self.model.Q_prior)
        logdet_Q_prior: float = self.solver.logdet()

        log_prior_latent_parameters: float = (
            + 0.5 * logdet_Q_prior
        )

        if x is not None:
            log_prior_latent_parameters -= 0.5 * x.T @ self.model.Q_prior @ x

        return log_prior_latent_parameters

    def _evaluate_conditional_latent_parameters(
        self,
        Q_conditional: NDArray,
        x: NDArray = None,
        x_mean: NDArray = None,
    ) -> float:
        """Evaluation of the conditional of the latent parameters at x using
        the conditional precision matrix Q_conditional and the mean x_mean.

        Parameters
        ----------
        Q_conditional : NDArray
            Conditional precision matrix.
        x : NDArray
            Latent parameters.
        x_mean : NDArray
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
        """
        # Compute the log determinant of Q_conditional
        logdet_Q_conditional = self.solver.logdet()

        # Compute the quadratic form (x - x_mean).T @ Q_conditional @ (x - x_mean)
        if x is None and x_mean is not None:
            quadratic_form = x_mean.T @ Q_conditional @ x_mean
        elif x is None and x_mean is None:
            quadratic_form = 0.0
        else:
            quadratic_form = (x - x_mean).T @ Q_conditional @ (x - x_mean)
        
        # Compute the log conditional
        log_conditional = (
            0.5 * logdet_Q_conditional - 0.5 * quadratic_form
        )

        return log_conditional