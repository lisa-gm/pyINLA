# Copyright 2024 pyINLA authors. All rights reserved.

import math
import time

from scipy.optimize import minimize

from pyinla import ArrayLike, comm_rank, comm_size, sparse, xp
from pyinla.core.pyinla_config import PyinlaConfig

from pyinla.core.model import Model


from pyinla.solvers.scipy_solver import ScipySolver
from pyinla.solvers.serinv_solver import SerinvSolver
from pyinla.utils.gpu import set_device
from pyinla.utils.multiprocessing import allreduce, bcast, print_msg, synchronize
from pyinla.utils.mapping import theta_array2dict, theta_dict2array


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

        # --- Load design matrix
        self.a = sparse.load_npz(pyinla_config.input_dir / "a.npz")
        self.n_latent_parameters = self.a.shape[1]

        # --- Load latent parameters vector
        try:
            self.x = xp.load(pyinla_config.input_dir / "x.npy")
        except FileNotFoundError:
            self.x = xp.ones((self.a.shape[1]), dtype=float)

        self._check_dimensions()

        self.sparsity_Q_conditional = "bta"
        self.sparsity_Q_prior = "bt"

        # --- Initialize model
        self.model = Model(pyinla_config=self.pyinla_config)

        # --- Initialize solver
        if self.pyinla_config.solver.type == "scipy":
            self.solver = ScipySolver(
                pyinla_config, self.model.ns, self.model.nb, self.model.nt
            )
        elif self.pyinla_config.solver.type == "serinv":
            self.solver = SerinvSolver(
                pyinla_config, self.model.ns, self.model.nb, self.model.nt
            )

        # --- Initialize theta
        self.theta_initial: ArrayLike = theta_dict2array(
            self.model.get_theta(), self.likelihood.get_theta()
        )
        self.dim_theta = len(self.theta_initial)
        self.theta = self.theta_initial
        self.gradient_f = xp.zeros(self.dim_theta)

        # --- Metrics
        self.f_values: list[float] = []
        self.theta_values: list[ArrayLike] = []

        # --- Set up recurrent variables
        self.Q_conditional: sparse.sparray = None

        print_msg("INLA initialized.", flush=True)
        print_msg("Model:", self.pyinla_config.model.type, flush=True)
        if len(self.model.get_theta()) > 0:
            print_msg(
                "Prior hyperparameters model:",
                self.pyinla_config.prior_hyperparameters.type,
                flush=True,
            )
            print_msg(
                f"  Prior theta model - spatial range. mean : {self.prior_hyperparameters.mean_theta_spatial_range}, precision : {self.prior_hyperparameters.precision_theta_spatial_range}\n"
                f"  Prior theta model - temporal range. mean : {self.prior_hyperparameters.mean_theta_temporal_range}, precision : {self.prior_hyperparameters.precision_theta_temporal_range}\n"
                f"  Prior theta model - spatio-temporal variation. mean : {self.prior_hyperparameters.mean_theta_spatio_temporal_variation}, precision : {self.prior_hyperparameters.precision_theta_spatio_temporal_variation}",
                flush=True,
            )

        if len(self.likelihood.get_theta()) > 0:
            print_msg(
                "   Prior hyperparameters likelihood:",
                self.pyinla_config.prior_hyperparameters.type,
                flush=True,
            )
            print_msg(
                f"  Prior theta likelihood. Mean : {self.prior_hyperparameters.mean_theta_observations}, precision : {self.prior_hyperparameters.precision_theta_observations}\n",
                flush=True,
            )
            print_msg("   Initial theta:", self.theta_initial, flush=True)
        print_msg("   Likelihood:", self.pyinla_config.likelihood.type, flush=True)

    def run(self) -> ArrayLike:
        """Fit the model using INLA."""

        # maxiter = 3
        # for i in range(maxiter):
        #     tic = time.perf_counter()
        #     self.f_value, self.gradient_f = self._objective_function(self.theta_initial)
        #     toc = time.perf_counter()
        #     print_msg(
        #         f"i: {i}. MPI size: {comm_size}. Time objective function call: {toc - tic} s. f value: {self.f_value}",
        #         flush=True,
        #     )
        #     self.theta_initial = self.theta_initial + 0.1

        if len(self.theta) > 0:
            # grad_f_init = self._evaluate_gradient_f(self.theta_initial)
            # print_msg(f"Initial gradient: {grad_f_init}", flush=True)

            tic = time.perf_counter()
            result = minimize(
                self._objective_function,
                self.theta_initial,
                method="BFGS",
                jac=self.pyinla_config.minimize.jac,
                options={
                    "maxiter": self.pyinla_config.minimize.max_iter,
                    "gtol": self.pyinla_config.minimize.gtol,
                    "c1": self.pyinla_config.minimize.c1,
                    "c2": self.pyinla_config.minimize.c2,
                    "disp": self.pyinla_config.minimize.disp,
                },
            )
            toc = time.perf_counter()
            print_msg(f"Total time minimize: {toc - tic} s", flush=True)

            if result.success:
                print_msg(
                    "Optimization converged successfully after",
                    result.nit,
                    "iterations.",
                    flush=True,
                )
                self.theta = result.x
                theta_model, theta_likelihood = theta_array2dict(
                    self.theta, self.model.get_theta(), self.likelihood.get_theta()
                )
                # print_msg("Optimal theta:", self.theta_star)
                # print_msg("Latent parameters:", self.x)
            else:
                print_msg("Optimization did not converge.", flush=True)
                return False

        # only run inner iteration
        else:
            print("in evaluate f.")
            self.f_value = self._evaluate_f(self.theta_initial)

        # print_msg("counter:", self.counter, flush=True)

        # TODO: check that Q_conditional was constructed using the right theta
        # if comm_rank == 0:
        #     if self.Q_conditional is not None:
        #         self._placeholder_marginals_latent_parameters(self.Q_conditional)
        #     else:
        #         print_msg("Q_conditional not defined.", flush=True)
        #         raise ValueError

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

    def _objective_function(self, theta_i: ArrayLike) -> float:
        """Objective function for optimization."""

        t_objective_function = time.perf_counter()
        # generate theta matrix with different theta's to evaluate
        # currently central difference scheme is used for gradient
        number_f_evaluations = 2 * self.dim_theta + 1
        f_values_local = xp.zeros(number_f_evaluations)

        main_eval_list = xp.zeros(number_f_evaluations, dtype=bool)
        main_eval_list[0] = True

        # task to rank assignment
        task_to_rank = xp.zeros(number_f_evaluations, dtype=int)

        for i in range(number_f_evaluations):
            task_to_rank[i] = i % comm_size

        # print_msg("task_to_rank: ", task_to_rank, flush=True)

        # TODO: eps mat constant, can live outside. size of theta_mat also constant, preallocate before?
        epsMat = self.eps_gradient_f * xp.eye(self.dim_theta)
        theta_mat = xp.repeat(theta_i.reshape(-1, 1), number_f_evaluations, axis=1)
        # store f_theta_i, f_theta_plus, f_theta_minus
        theta_mat[:, 1 : 1 + self.dim_theta] += epsMat
        theta_mat[:, self.dim_theta + 1 : number_f_evaluations] -= epsMat
        # print_msg("theta_mat: \n", theta_mat)

        for i in range(number_f_evaluations):
            if task_to_rank[i] == comm_rank:
                # print("Rank: ", comm_rank, "i: ", i, "main_eval: ", main_eval_list[i])
                t_f_eval = time.perf_counter()
                f_values_local[i] = self._evaluate_f(
                    theta_mat[:, i], main_eval=main_eval_list[i]
                )
                t_f_eval = time.perf_counter() - t_f_eval
                # print(
                #     f"Rank: {comm_rank}, i: {i}, f_value: {f_values_local[i]}, time: {t_f_eval}",
                #     flush=True,
                # )

        synchronize()

        # gather f_values from all ranks using MPI_Allreduce with MPI_SUM
        f_values = xp.zeros(number_f_evaluations)
        allreduce(f_values_local, f_values, op="sum")

        # print_msg("f_values: ", f_values, flush=True)

        f_theta = f_values[0]

        # compute gradient using central difference scheme
        gradient_f_theta = xp.zeros(self.dim_theta)
        for i in range(self.dim_theta):
            gradient_f_theta[i] = (
                f_values[i + 1] - f_values[i + self.dim_theta + 1]
            ) / (2 * self.eps_gradient_f)

        # tic = time.perf_counter()
        # gradient_f_theta_old = xp.zeros(self.dim_theta)

        # # TODO: Theses evaluations are independant and can be performed
        # # in parallel.
        # for i in range(self.dim_theta):
        #     theta_plus = theta_i.copy()
        #     theta_minus = theta_i.copy()

        #     theta_plus[i] += self.eps_gradient_f
        #     theta_minus[i] -= self.eps_gradient_f

        #     f_plus = self._evaluate_f(theta_plus)
        #     f_minus = self._evaluate_f(theta_minus)

        #     print_msg(
        #         f"NEW:: i: {i}, f_forward: {f_values[i + 1]}, f_backward: {f_values[i + self.dim_theta + 1]}"
        #     )
        #     print_msg(f"OLD:: i: {i}, f_forward: {f_plus}, f_backward: {f_minus}")
        #     print("diff forward: ", f_values[i + 1] - f_plus)
        #     print("diff backward: ", f_values[i + self.dim_theta + 1] - f_minus)

        #     gradient_f_theta_old[i] = (f_plus - f_minus) / (2 * self.eps_gradient_f)
        # toc = time.perf_counter()
        # print_msg("   evaluate_gradient_f time:", toc - tic, flush=True)

        # print_msg(f"Gradient: {gradient_f_theta}", flush=True)
        # print_msg(f"Gradient old: {gradient_f_theta_old}", flush=True)

        # broadcast self.x from current theta from rank 0 to all other ranks for next iteration

        t_mpi_bcast = time.perf_counter()
        synchronize()
        bcast(self.x, root=0)

        synchronize()

        t_mpi_bcast = time.perf_counter() - t_mpi_bcast
        # print_msg(f"MPI_Bcast time: {t_mpi_bcast}", flush=True)

        t_objective_function = time.perf_counter() - t_objective_function
        theta_i_str = ", ".join([f"{theta:.3f}" for theta in theta_i])
        print_msg(
            f"theta: [{theta_i_str}], Function value: {f_theta:.3f}, time: {t_objective_function:.3f}",
            flush=True,
        )
        return (f_theta, gradient_f_theta)

    def _evaluate_f(self, theta_i: ArrayLike, main_eval: bool = True) -> float:
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

        # print_msg("Evaluate f()", flush=True)

        self.theta = theta_i

        theta_model, theta_likelihood = theta_array2dict(
            theta_i, self.model.get_theta(), self.likelihood.get_theta()
        )

        # --- Evaluate the log prior of the hyperparameters
        # tic = time.perf_counter()
        log_prior_hyperparameters = self.prior_hyperparameters.evaluate_log_prior(
            theta_model, theta_likelihood
        )
        # toc = time.perf_counter()
        # print_msg("log prior hyperparameters: ", log_prior_hyperparameters)
        # print_msg("   (1/6) evaluate_log_prior time:", toc - tic, flush=True)

        # --- Construct the prior precision matrix of the latent parameters
        # tic = time.perf_counter()
        Q_prior = self.model.construct_Q_prior(theta_model)
        # toc = time.perf_counter()
        # print_msg("   (2/6) construct_Q_prior time:", toc - tic, flush=True)

        # --- Optimize x (latent parameters) and construct conditional precision matrix
        x_local = xp.copy(self.x)
        # tic = time.perf_counter()
        self.Q_conditional, x_local, logdet_Q_conditional = self._inner_iteration(
            Q_prior, x_local, theta_likelihood
        )
        # toc = time.perf_counter()
        # print_msg("   (3/6) inner_iteration time:", toc - tic, flush=True)
        # print(f"rank: {comm_rank}. after inner iteration x: ", self.x[:10])

        # assign initial guess for x for next iteration
        if main_eval:
            # print(f"rank: {comm_rank}. Assigning x_local to x")
            self.x = x_local

        # --- Evaluate likelihood at the optimized latent parameters x_star
        # tic = time.perf_counter()
        eta = self.a @ x_local
        likelihood = self.likelihood.evaluate_likelihood(eta, self.y, theta_likelihood)
        # toc = time.perf_counter()
        # print_msg("   (4/6) evaluate_likelihood time:", toc - tic, flush=True)
        # print_msg("likelihood: ", likelihood)

        # --- Evaluate the conditional of the latent parameters at x_star
        # tic = time.perf_counter()
        conditional_latent_parameters = self._evaluate_conditional_latent_parameters(
            self.Q_conditional, x_local, x_local, logdet_Q_conditional
        )
        # toc = time.perf_counter()
        # print_msg(
        #     "   (6/6) evaluate_conditional_latent_parameters time:",
        #     toc - tic,
        #     flush=True,
        # )
        # print_msg("conditional latent parameters: ", conditional_latent_parameters)

        # --- Evaluate the prior of the latent parameters at x_star
        # tic = time.perf_counter()
        prior_latent_parameters = self._evaluate_prior_latent_parameters(
            Q_prior, x_local
        )
        # toc = time.perf_counter()
        # print_msg(
        #     "   (5/6) evaluate_prior_latent_parameters time:", toc - tic, flush=True
        # )
        # print_msg("prior latent parameters: ", prior_latent_parameters)

        f_theta = -1 * (
            log_prior_hyperparameters
            + likelihood
            + prior_latent_parameters
            - conditional_latent_parameters
        )

        # print_msg(f"theta: {theta_i},      Function value: {f_theta}")

        if f_theta < self.min_f:
            self.min_f = f_theta
            self.counter += 1
            # print(f"theta: {theta_i},      Function value: {f_theta}", flush=True)
            # print_msg(f"Minimum function value: {self.min_f}. Counter: {self.counter}")

        return f_theta

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

        tic = time.perf_counter()
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
        toc = time.perf_counter()
        print_msg("   evaluate_gradient_f time:", toc - tic, flush=True)

        print_msg(f"Gradient: {grad_f}", flush=True)

        return grad_f

    def _inner_iteration(
        self, Q_prior: sparse.sparray, x_i: ArrayLike, theta_likelihood: dict
    ):
        x_update = xp.zeros_like(x_i)
        x_i_norm = 1

        # print_msg("   Starting inner iteration", flush=True)
        # print(f"In inner iteration: Rank {comm_rank} x: {x_i[:10]}")

        counter = 0
        while x_i_norm >= self.eps_inner_iteration:
            if counter > self.inner_iteration_max_iter:
                print_msg("current theta value: ", self.theta)
                raise ValueError(
                    f"Inner iteration did not converge after {counter} iterations."
                )
            # print_msg(f"      inner iteration {counter} norm: {x_i_norm}", flush=True)

            # tic = time.perf_counter()
            x_i[:] += x_update[:]
            eta = self.a @ x_i
            # print_msg("eta: ", eta[:6])

            # TODO: need to vectorize !!
            # gradient_likelihood = gradient_finite_difference_5pt(
            #     self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
            # )
            # with time_range('computeGradientLik', color_id=0):
            gradient_likelihood = self.likelihood.evaluate_gradient_likelihood(
                eta, self.y, theta_likelihood
            )
            # print_msg("gradient_likelihood: ", gradient_likelihood[:6])
            # toc = time.perf_counter()
            # print_msg("         evaluate_likelihood time:", toc - tic, flush=True)

            # tic = time.perf_counter()
            # with time_range('constructRhs', color_id=0):
            rhs = -1 * Q_prior @ x_i + self.a.T @ gradient_likelihood

            # TODO: need to vectorize
            # hessian_likelihood_diag = hessian_diag_finite_difference_5pt(
            #     self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
            # )
            # hessian_likelihood = diags(hessian_likelihood_diag)
            # with time_range('computeHessianLik', color_id=0):
            hessian_likelihood = self.likelihood.evaluate_hessian_likelihood(
                eta, self.y, theta_likelihood
            )
            # print("hessian_likelihood: ", hessian_likelihood.diagonal()[:6])
            # toc = time.perf_counter()
            # print_msg(
            #     "         hessian_diag_finite_difference_5pt time:",
            #     toc - tic,
            #     flush=True,
            # )

            # tic = time.perf_counter()
            Q_conditional = self.model.construct_Q_conditional(
                Q_prior,
                self.a,
                hessian_likelihood,
            )
            # toc = time.perf_counter()
            # print_msg("         construct_Q_conditional time:", toc - tic, flush=True)

            # tic = time.perf_counter()
            self.solver.cholesky(Q_conditional, sparsity=self.sparsity_Q_conditional)
            # toc = time.perf_counter()
            # print_msg("         Solver Call Q_conditional time:", toc - tic, flush=True)

            # tic = time.perf_counter()
            x_update[:] = self.solver.solve(rhs, sparsity=self.sparsity_Q_conditional)
            # toc = time.perf_counter()
            # print_msg("         solve Q_conditional time:", toc - tic, flush=True)

            # with time_range('computeNorm', color_id=0):
            x_i_norm = xp.linalg.norm(x_update)

            counter += 1
            # print_msg(f"Inner iteration {counter} norm: {x_i_norm}")

        logdet = self.solver.logdet()

        # print_msg("Inner iteration converged after", counter, "iterations.")
        return Q_conditional, x_i, logdet

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
            self.theta, self.model.get_theta(), self.likelihood.get_theta()
        )
        if self.model.get_theta() != theta_model:
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
