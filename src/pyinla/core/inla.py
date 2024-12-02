# Copyright 2024 pyINLA authors. All rights reserved.

import math
import time

from scipy.optimize import minimize

from pyinla import ArrayLike, comm_rank, comm_size, sparse, xp
from pyinla.core.pyinla_config import PyinlaConfig

from pyinla.core.model import Model


from pyinla.solvers.sparse_solver import SparseSolver
from pyinla.solvers.structured_solver import SerinvSolver
from pyinla.utils.gpu import set_device
from pyinla.utils.multiprocessing import allreduce, bcast, print_msg, synchronize


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

        self.inner_iteration_max_iter = self.pyinla_config.inner_iteration_max_iter
        self.eps_inner_iteration = self.pyinla_config.eps_inner_iteration
        self.eps_gradient_f = self.pyinla_config.eps_gradient_f

        # --- Configure HPC
        set_device(comm_rank)

        self.sparsity_Q_conditional = "bta"
        self.sparsity_Q_prior = "bt"

        # --- Initialize model
        self.model = Model(pyinla_config=self.pyinla_config)

        # --- Initialize solver
        if self.pyinla_config.solver.type == "scipy":
            self.solver = SparseSolver(
                pyinla_config, self.model.ns, self.model.nb, self.model.nt
            )
        elif self.pyinla_config.solver.type == "serinv":
            self.solver = SerinvSolver(
                pyinla_config, self.model.ns, self.model.nb, self.model.nt
            )

        # --- Set up recurrent variables
        self.gradient_f = xp.zeros(self.model.n_hyperparameters)
        self.f_values_i = xp.zeros(2 * self.model.n_hyperparameters + 1)
        self.x_central = xp.zeros_like(self.model.x)
        self.x_local = xp.zeros_like(self.model.x)

        # --- Metrics
        self.f_values: list[float] = []
        self.theta_values: list[ArrayLike] = []

        print_msg("INLA initialized.", flush=True)

    def run(self) -> ArrayLike:
        """Fit the model using INLA."""

        if len(self.model.theta) > 0:

            def callback(xk):
                self.theta_values.append(xk.x.copy())
                self.f_values.append(self.fun.copy())

            result = minimize(
                self._objective_function,
                self.model.theta,
                method="BFGS",
                jac=self.pyinla_config.minimize.jac,
                options={
                    "maxiter": self.pyinla_config.minimize.max_iter,
                    "gtol": self.pyinla_config.minimize.gtol,
                    "c1": self.pyinla_config.minimize.c1,
                    "c2": self.pyinla_config.minimize.c2,
                    "disp": self.pyinla_config.minimize.disp,
                },
                callback=callback,
            )

            if result.success:
                print_msg(
                    "Optimization converged successfully after",
                    result.nit,
                    "iterations.",
                    flush=True,
                )
            else:
                print_msg("Optimization did not converge.", flush=True)
                return False

        # only run inner iteration
        else:
            print("in evaluate f.")
            self.f_value = self._evaluate_f(self.model.theta)

        return True

    def _objective_function(self, theta_i: ArrayLike) -> float:
        """Objective function for optimization."""

        # generate theta matrix with different theta's to evaluate
        # currently central difference scheme is used for gradient
        number_f_evaluations = 2 * self.model.n_hyperparameters + 1
        f_values_local = xp.zeros(number_f_evaluations)

        # task to rank assignment
        task_to_rank = xp.zeros(number_f_evaluations, dtype=int)

        for i in range(number_f_evaluations):
            task_to_rank[i] = i % comm_size

        # TODO: eps mat constant, can live outside. size of theta_mat also constant, preallocate before?
        epsMat = self.eps_gradient_f * xp.eye(self.model.n_hyperparameters)
        theta_mat = xp.repeat(theta_i.reshape(-1, 1), number_f_evaluations, axis=1)
        # store f_theta_i, f_theta_plus, f_theta_minus
        theta_mat[:, 1 : 1 + self.model.n_hyperparameters] += epsMat
        theta_mat[:, self.model.n_hyperparameters + 1 : number_f_evaluations] -= epsMat

        for i in range(number_f_evaluations):
            if task_to_rank[i] == comm_rank:
                f_values_local[i], x_i = self._evaluate_f(theta_mat[:, i])

                if i == 0:
                    self.x_central[:] = x_i

        synchronize()
        bcast(self.x_central, root=0)

        # gather f_values from all ranks using MPI_Allreduce with MPI_SUM
        self.f_values_i[:] = 0.0
        allreduce(f_values_local, self.f_values_i, op="sum")

        # compute gradient using central difference scheme
        for i in range(self.model.n_hyperparameters):
            self.gradient_f[i] = (
                self.f_values_i[i + 1]
                - self.f_values_i[i + self.model.n_hyperparameters + 1]
            ) / (2 * self.eps_gradient_f)

        synchronize()

        theta_i_str = ", ".join([f"{theta:.3f}" for theta in theta_i])
        print_msg(
            f"theta: [{theta_i_str}], Function value: {self.f_values_i[0]:.3f}",
            flush=True,
        )
        return (self.f_values_i[0], self.gradient_f)

    def _evaluate_f(self, theta_i: ArrayLike) -> float:
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

        self.model.theta = theta_i

        # --- Evaluate the log prior of the hyperparameters
        log_prior_hyperparameters = self.prior_hyperparameters.evaluate_log_prior(
            theta_model, theta_likelihood
        )

        # --- Construct the prior precision matrix of the latent parameters
        Q_prior = self.model.construct_Q_prior(theta_model)

        # --- Optimize x (latent parameters) and construct conditional precision matrix
        self.x_local[:] = xp.copy(self.x)
        logdet_Q_conditional = self._inner_iteration(self.x_local)

        # --- Evaluate likelihood at the optimized latent parameters x_star
        eta = self.a @ self.x_local
        likelihood = self.likelihood.evaluate_likelihood(eta, self.y, theta_likelihood)

        # --- Evaluate the conditional of the latent parameters at x_star
        conditional_latent_parameters = self._evaluate_conditional_latent_parameters(
            self.Q_conditional, self.x_local, self.x_local, logdet_Q_conditional
        )

        # --- Evaluate the prior of the latent parameters at x_star
        prior_latent_parameters = self._evaluate_prior_latent_parameters(
            Q_prior, self.x_local
        )

        f_theta = -1 * (
            log_prior_hyperparameters
            + likelihood
            + prior_latent_parameters
            - conditional_latent_parameters
        )

        if f_theta < self.min_f:
            self.min_f = f_theta
            self.counter += 1

        return f_theta, self.x_local

    def _evaluate_gradient_f(self, theta_i: ArrayLike) -> ArrayLike:
        """evaluate the gradient of the objective function f(theta) = log(p(theta|y)).

        Notes
        -----
        Evaluate the gradient of the objective function f(theta) = log(p(theta|y)) wrt to theta
        using a finite difference approximation. For now implement only central difference scheme.

        Returns
        -------
        grad_f : ArrayLike
            Gradient of the objective function f(theta) evaluated at theta_i.

        """

        print_msg("Evaluate gradient_f()", flush=True)

        dim_theta = theta_i.shape[0]
        grad_f = xp.zeros(dim_theta)

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
        print_msg("   evaluate_gradient_f time:", toc - tic, flush=True)

        print_msg(f"Gradient: {grad_f}", flush=True)

        return grad_f

    def _inner_iteration(
        self,
        x_i: ArrayLike,
    ):
        x_update = xp.zeros_like(x_i)
        x_i_norm = 1

        counter = 0
        while x_i_norm >= self.eps_inner_iteration:
            if counter > self.inner_iteration_max_iter:
                print_msg("current theta value: ", self.model.theta)
                raise ValueError(
                    f"Inner iteration did not converge after {counter} iterations."
                )

            x_i[:] += x_update[:]
            eta = self.model.a @ x_i

            Q_conditional = self.model.construct_Q_conditional(
                eta,
            )
            self.solver.cholesky(Q_conditional, sparsity=self.sparsity_Q_conditional)

            rhs = self.model.construct_information_vector(eta, x_i)
            x_update[:] = self.solver.solve(rhs, sparsity=self.sparsity_Q_conditional)

            x_i_norm = xp.linalg.norm(x_update)
            counter += 1

        logdet = self.solver.logdet()

        # print_msg("Inner iteration converged after", counter, "iterations.")
        return logdet

    def _evaluate_prior_latent_parameters(self, Q_prior: sparse.sparray, x: ArrayLike):
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

        # with time_range('callCholesky', color_id=0):
        self.solver.cholesky(Q_prior, sparsity=self.sparsity_Q_prior)

        # with time_range('getLogDet', color_id=0):
        logdet_Q_prior = self.solver.logdet()

        # with time_range('compute_xtQx', color_id=0):
        log_prior_latent_parameters = (
            -n / 2 * xp.log(2 * math.pi)
            + 0.5 * logdet_Q_prior
            - 0.5 * x.T @ Q_prior @ x
        )

        return log_prior_latent_parameters

    def _evaluate_conditional_latent_parameters(
        self, Q_conditional, x, x_mean, logdet_Q_conditional
    ):
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

        # TODO: not actually needed. already computed.
        # get current theta, check if this theta matches the theta used to construct Q_conditional
        # if yes, check if L already computed, if yes -> takes this L
        # self.solver.cholesky(Q_conditional, sparsity="bta")
        # logdet_Q_conditional = self.solver.logdet()
        # print_msg("in evaluate conditional latent. logdet: ", logdet_Q_conditional)

        # TODO: evaluate it at the mean -> this becomes zero ...
        # log_conditional_latent_parameters = (
        #     -n / 2 * xp.log(2 * math.pi)
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
            self.model.theta, self.model.theta, self.likelihood.get_theta()
        )
        if self.model.theta != theta_model:
            print_msg("theta of Q_conditional does not match current theta")
            # raise ValueError

        self.solver.full_inverse()
        Q_inverse_selected = self.solver.get_selected_inverse()

        # min_size = min(self.n_latent_parameters, 6)
        # print_msg(f"Q_inverse_selected[:{min_size}, :{min_size}]: \n", Q_inverse_selected[:min_size, :min_size].toarray())

        latent_parameters_marginal_sd = xp.sqrt(Q_inverse_selected.diagonal())
        print_msg(
            f"standard deviation fixed effects: {latent_parameters_marginal_sd[-self.pyinla_config.model.n_fixed_effects:]}"
        )

        return Q_inverse_selected
