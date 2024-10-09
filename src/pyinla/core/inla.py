# Copyright 2024 pyINLA authors. All rights reserved.

import math
import os
import time

import numpy as np
from mpi4py import MPI
from numpy.typing import ArrayLike

# from scipy.optimize import minimize
from scipy.sparse import load_npz, sparray  # diags,

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
from pyinla.solvers.cusparse_solver import CuSparseSolver
from pyinla.solvers.scipy_solver import ScipySolver
from pyinla.solvers.serinv_solver import SerinvSolverCPU

# from pyinla.utils.finite_difference_stencils import (
#     gradient_finite_difference_5pt,
#     hessian_diag_finite_difference_5pt,
# )
from pyinla.utils.other_utils import print_mpi
from pyinla.utils.theta_utils import theta_array2dict, theta_dict2array

# try:
# from cupy.cuda import nvtx
from cupyx.profiler import time_range

# CUDA_AVAIL = True
# except:
#     CUDA_AVAIL = False
#     pass


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


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

        self._check_dimensions()

        # --- Initialize model
        if self.pyinla_config.model.type == "regression":
            self.model = RegressionModel(pyinla_config, self.n_latent_parameters)
            print_mpi("Regression model initialized.")
        elif self.pyinla_config.model.type == "spatio-temporal":
            self.model = SpatioTemporalModel(pyinla_config, self.n_latent_parameters)
            print_mpi("Spatio-temporal model initialized.")
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
            print_mpi("Gaussian likelihood initialized.")
        elif self.pyinla_config.likelihood.type == "poisson":
            self.likelihood = PoissonLikelihood(pyinla_config, self.n_observations)
            print_mpi("Poisson likelihood initialized.")
        elif self.pyinla_config.likelihood.type == "binomial":
            self.likelihood = BinomialLikelihood(pyinla_config, self.n_observations)
            print_mpi("Binomial likelihood initialized.")
        else:
            raise ValueError(
                f"Likelihood '{self.pyinla_config.likelihood.type}' not implemented."
            )

        # --- print MPI information
        print_mpi(f"MPI rank: {comm_rank}, MPI size: {comm_size}", flush=True)

        # --- Initialize solver
        num_threads = os.getenv("OMP_NUM_THREADS")
        print_mpi(f"OMP_NUM_THREADS: {num_threads}")

        if self.pyinla_config.solver.type == "scipy":
            self.solver_Q_prior = ScipySolver(pyinla_config)
            self.solver_Q_conditional = ScipySolver(pyinla_config)
            print_mpi("Scipy solver initialized.")
        elif self.pyinla_config.solver.type == "cusparse":
            self.solver_Q_prior = CuSparseSolver(pyinla_config)
            self.solver_Q_conditional = CuSparseSolver(pyinla_config)
        elif self.pyinla_config.solver.type == "serinv_cpu":
            if self.pyinla_config.model.type == "regression":
                raise ValueError(
                    f"Solver '{self.pyinla_config.solver.type}' not implemented for regression model."
                )
            else:
                self.solver_Q_prior = SerinvSolverCPU(
                    pyinla_config, self.model.ns, self.model.nb, self.model.nt
                )
                self.solver_Q_conditional = SerinvSolverCPU(
                    pyinla_config, self.model.ns, self.model.nb, self.model.nt
                )
                print_mpi("Serinv CPU solver initialized.")
        else:
            raise ValueError(
                f"Solver '{self.pyinla_config.solver.type}' not implemented."
            )

        # --- Initialize theta
        self.theta_initial: ArrayLike = theta_dict2array(
            self.model.get_theta(), self.likelihood.get_theta()
        )
        self.dim_theta = len(self.theta_initial)

        self.theta = self.theta_initial
        self.f_value = 1e10
        self.gradient_f = np.zeros(self.dim_theta)

        self.counter = 0
        self.min_f = 1e10

        # --- Set up recurrent variables
        self.Q_conditional: sparray = None

        print_mpi("INLA initialized.", flush=True)
        print_mpi("Model:", self.pyinla_config.model.type, flush=True)
        if len(self.model.get_theta()) > 0:
            print_mpi(
                "Prior hyperparameters model:",
                self.pyinla_config.prior_hyperparameters.type,
                flush=True,
            )
            print_mpi(
                f"  Prior theta model - spatial range. mean : {self.prior_hyperparameters.mean_theta_spatial_range}, precision : {self.prior_hyperparameters.precision_theta_spatial_range}\n"
                f"  Prior theta model - temporal range. mean : {self.prior_hyperparameters.mean_theta_temporal_range}, precision : {self.prior_hyperparameters.precision_theta_temporal_range}\n"
                f"  Prior theta model - spatio-temporal variation. mean : {self.prior_hyperparameters.mean_theta_spatio_temporal_variation}, precision : {self.prior_hyperparameters.precision_theta_spatio_temporal_variation}",
                flush=True,
            )

        if len(self.likelihood.get_theta()) > 0:
            print_mpi(
                "   Prior hyperparameters likelihood:",
                self.pyinla_config.prior_hyperparameters.type,
                flush=True,
            )
            print_mpi(
                f"  Prior theta likelihood. Mean : {self.prior_hyperparameters.mean_theta_observations}, precision : {self.prior_hyperparameters.precision_theta_observations}\n",
                flush=True,
            )
            print_mpi("   Initial theta:", self.theta_initial, flush=True)
        print_mpi("   Likelihood:", self.pyinla_config.likelihood.type, flush=True)

    def run(self) -> np.ndarray:
        """Fit the model using INLA."""

        tic = time.perf_counter()
        self.f_value, self.gradient_f = self._objective_function(self.theta_initial)
        toc = time.perf_counter()
        print_mpi(
            f"MPI size: {comm_size}. Time objective function call: {toc - tic} s. f value: {self.f_value}",
            flush=True,
        )

        # if len(self.theta) > 0:
        #     # grad_f_init = self._evaluate_gradient_f(self.theta_initial)
        #     # print_mpi(f"Initial gradient: {grad_f_init}", flush=True)

        #     tic = time.perf_counter()
        #     result = minimize(
        #         self._objective_function,
        #         self.theta_initial,
        #         method="BFGS",
        #         jac=True,
        #         options={
        #             "maxiter": self.minimize_max_iter,
        #             # "maxiter": 1,
        #             "gtol": 1e-1,
        #             "disp": False,
        #         },
        #     )
        #     toc = time.perf_counter()
        #     print_mpi(f"1-Minimize iteration time: {toc - tic} s", flush=True)

        #     if result.success:
        #         print_mpi(
        #             "Optimization converged successfully after",
        #             result.nit,
        #             "iterations.",
        #             flush=True,
        #         )
        #         self.theta = result.x
        #         theta_model, theta_likelihood = theta_array2dict(
        #             self.theta, self.model.get_theta(), self.likelihood.get_theta()
        #         )
        #         # print_mpi("Optimal theta:", self.theta_star)
        #         # print_mpi("Latent parameters:", self.x)
        #     else:
        #         print_mpi("Optimization did not converge.", flush=True)
        #         return False

        # print_mpi("counter:", self.counter, flush=True)

        # TODO: check that Q_conditional was constructed using the right theta
        # if comm_rank == 0:
        #     if self.Q_conditional is not None:
        #         self._placeholder_marginals_latent_parameters(self.Q_conditional)
        #     else:
        #         print_mpi("Q_conditional not defined.", flush=True)
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

    @time_range()
    def _objective_function(self, theta_i: np.ndarray) -> float:
        """Objective function for optimization."""

        t_objective_function = time.perf_counter()
        # generate theta matrix with different theta's to evaluate
        # currently central difference scheme is used for gradient
        number_f_evaluations = 2 * self.dim_theta + 1
        f_values_local = np.zeros(number_f_evaluations)

        main_eval_list = np.zeros(number_f_evaluations, dtype=bool)
        main_eval_list[0] = True

        # task to rank assignment
        task_to_rank = np.zeros(number_f_evaluations, dtype=int)

        for i in range(number_f_evaluations):
            task_to_rank[i] = i % comm_size

        # print_mpi("task_to_rank: ", task_to_rank, flush=True)

        # TODO: eps mat constant, can live outside. size of theta_mat also constant, preallocate before?
        epsMat = self.eps_gradient_f * np.eye(self.dim_theta)
        theta_mat = np.repeat(theta_i.reshape(-1, 1), number_f_evaluations, axis=1)
        # store f_theta_i, f_theta_plus, f_theta_minus
        theta_mat[:, 1 : 1 + self.dim_theta] += epsMat
        theta_mat[:, self.dim_theta + 1 : number_f_evaluations] -= epsMat
        # print_mpi("theta_mat: \n", theta_mat)

        for i in range(number_f_evaluations):
            if task_to_rank[i] == comm_rank:
                # print("Rank: ", comm_rank, "i: ", i, "main_eval: ", main_eval_list[i])
                t_f_eval = time.perf_counter()
                f_values_local[i] = self._evaluate_f(
                    theta_mat[:, i], main_eval=main_eval_list[i]
                )
                t_f_eval = time.perf_counter() - t_f_eval
                print(
                    f"Rank: {comm_rank}, i: {i}, f_value: {f_values_local[i]}, time: {t_f_eval}",
                    flush=True,
                )

        MPI.COMM_WORLD.Barrier()

        # gather f_values from all ranks using MPI_Allreduce with MPI_SUM
        f_values = np.zeros(number_f_evaluations)
        MPI.COMM_WORLD.Allreduce(f_values_local, f_values, op=MPI.SUM)

        # print_mpi("f_values: ", f_values, flush=True)

        f_theta = f_values[0]

        # compute gradient using central difference scheme
        gradient_f_theta = np.zeros(self.dim_theta)
        for i in range(self.dim_theta):
            gradient_f_theta[i] = (
                f_values[i + 1] - f_values[i + self.dim_theta + 1]
            ) / (2 * self.eps_gradient_f)

        # tic = time.perf_counter()
        # gradient_f_theta_old = np.zeros(self.dim_theta)

        # # TODO: Theses evaluations are independant and can be performed
        # # in parallel.
        # for i in range(self.dim_theta):
        #     theta_plus = theta_i.copy()
        #     theta_minus = theta_i.copy()

        #     theta_plus[i] += self.eps_gradient_f
        #     theta_minus[i] -= self.eps_gradient_f

        #     f_plus = self._evaluate_f(theta_plus)
        #     f_minus = self._evaluate_f(theta_minus)

        #     print_mpi(
        #         f"NEW:: i: {i}, f_forward: {f_values[i + 1]}, f_backward: {f_values[i + self.dim_theta + 1]}"
        #     )
        #     print_mpi(f"OLD:: i: {i}, f_forward: {f_plus}, f_backward: {f_minus}")
        #     print("diff forward: ", f_values[i + 1] - f_plus)
        #     print("diff backward: ", f_values[i + self.dim_theta + 1] - f_minus)

        #     gradient_f_theta_old[i] = (f_plus - f_minus) / (2 * self.eps_gradient_f)
        # toc = time.perf_counter()
        # print_mpi("   evaluate_gradient_f time:", toc - tic, flush=True)

        # print_mpi(f"Gradient: {gradient_f_theta}", flush=True)
        # print_mpi(f"Gradient old: {gradient_f_theta_old}", flush=True)

        # broadcast self.x from current theta from rank 0 to all other ranks for next iteration

        t_mpi_bcast = time.perf_counter()
        MPI.COMM_WORLD.Barrier()
        MPI.COMM_WORLD.Bcast(self.x, root=0)

        MPI.COMM_WORLD.Barrier()

        t_mpi_bcast = time.perf_counter() - t_mpi_bcast
        print_mpi(f"MPI_Bcast time: {t_mpi_bcast}", flush=True)

        t_objective_function = time.perf_counter() - t_objective_function
        print_mpi(f"Objective function time: {t_objective_function}", flush=True)

        return (f_theta, gradient_f_theta)

    @time_range()
    def _evaluate_f(self, theta_i: np.ndarray, main_eval: bool = True) -> float:
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

        # print_mpi("Evaluate f()", flush=True)

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
        # print_mpi("   (1/6) evaluate_log_prior time:", toc - tic, flush=True)

        # --- Construct the prior precision matrix of the latent parameters
        # tic = time.perf_counter()
        Q_prior = self.model.construct_Q_prior(theta_model)
        # toc = time.perf_counter()
        # print_mpi("   (2/6) construct_Q_prior time:", toc - tic, flush=True)

        # --- Optimize x (latent parameters) and construct conditional precision matrix
        x_local = np.copy(self.x)
        # tic = time.perf_counter()
        self.Q_conditional, x_local = self._inner_iteration(
            Q_prior, x_local, theta_likelihood
        )
        # toc = time.perf_counter()
        # print_mpi("   (3/6) inner_iteration time:", toc - tic, flush=True)
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
        # print_mpi("   (4/6) evaluate_likelihood time:", toc - tic, flush=True)

        # --- Evaluate the prior of the latent parameters at x_star
        # tic = time.perf_counter()
        prior_latent_parameters = self._evaluate_prior_latent_parameters(
            Q_prior, x_local
        )
        # toc = time.perf_counter()
        # print_mpi(
        #     "   (5/6) evaluate_prior_latent_parameters time:", toc - tic, flush=True
        # )

        # --- Evaluate the conditional of the latent parameters at x_star
        # tic = time.perf_counter()
        conditional_latent_parameters = self._evaluate_conditional_latent_parameters(
            self.Q_conditional, x_local, x_local
        )
        # toc = time.perf_counter()
        # print_mpi(
        #     "   (6/6) evaluate_conditional_latent_parameters time:",
        #     toc - tic,
        #     flush=True,
        # )

        f_theta = -1 * (
            log_prior_hyperparameters
            + likelihood
            + prior_latent_parameters
            - conditional_latent_parameters
        )

        # print_mpi(f"theta: {theta_i},      Function value: {f_theta}")

        if f_theta < self.min_f:
            self.min_f = f_theta
            self.counter += 1
            print_mpi(f"theta: {theta_i},      Function value: {f_theta}", flush=True)
            # print_mpi(f"Minimum function value: {self.min_f}. Counter: {self.counter}")

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

        print_mpi("Evaluate gradient_f()", flush=True)

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
        print_mpi("   evaluate_gradient_f time:", toc - tic, flush=True)

        print_mpi(f"Gradient: {grad_f}", flush=True)

        return grad_f

    @time_range()
    def _inner_iteration(
        self, Q_prior: sparray, x_i: ArrayLike, theta_likelihood: dict
    ):
        x_update = np.zeros_like(x_i)
        x_i_norm = 1

        # print_mpi("   Starting inner iteration", flush=True)
        # print(f"In inner iteration: Rank {comm_rank} x: {x_i[:10]}")

        counter = 0
        while x_i_norm >= self.eps_inner_iteration:
            if counter > self.inner_iteration_max_iter:
                raise ValueError(
                    f"Inner iteration did not converge after {counter} iterations."
                )
            # print_mpi(f"      inner iteration {counter} norm: {x_i_norm}", flush=True)

            # tic = time.perf_counter()
            x_i[:] += x_update[:]
            eta = self.a @ x_i

            # TODO: need to vectorize !!
            # gradient_likelihood = gradient_finite_difference_5pt(
            #     self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
            # )
            gradient_likelihood = self.likelihood.evaluate_gradient_likelihood(
                eta, self.y, theta_likelihood
            )
            # toc = time.perf_counter()
            # print_mpi("         evaluate_likelihood time:", toc - tic, flush=True)

            # tic = time.perf_counter()
            rhs = -1 * Q_prior @ x_i + self.a.T @ gradient_likelihood

            # TODO: need to vectorize
            # hessian_likelihood_diag = hessian_diag_finite_difference_5pt(
            #     self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
            # )
            # hessian_likelihood = diags(hessian_likelihood_diag)
            hessian_likelihood = self.likelihood.evaluate_hessian_likelihood(
                eta, self.y, theta_likelihood
            )
            # toc = time.perf_counter()
            # print_mpi(
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
            # print_mpi("         construct_Q_conditional time:", toc - tic, flush=True)

            # tic = time.perf_counter()
            self.solver_Q_conditional.cholesky(Q_conditional)
            # toc = time.perf_counter()
            # print_mpi("         cholesky time:", toc - tic, flush=True)

            # tic = time.perf_counter()
            x_update[:] = self.solver_Q_conditional.solve(rhs)
            # toc = time.perf_counter()
            # print_mpi("         solve time:", toc - tic, flush=True)

            x_i_norm = np.linalg.norm(x_update)
            counter += 1
            # print_mpi(f"Inner iteration {counter} norm: {x_i_norm}")

        # print_mpi("Inner iteration converged after", counter, "iterations.")
        return Q_conditional, x_i

    @time_range()
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

    @time_range()
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
            print_mpi("theta of Q_conditional does not match current theta")
            # raise ValueError

        self.solver_Q_conditional.full_inverse()
        Q_inverse_selected = self.solver_Q_conditional.get_selected_inverse()

        # min_size = min(self.n_latent_parameters, 6)
        # print_mpi(f"Q_inverse_selected[:{min_size}, :{min_size}]: \n", Q_inverse_selected[:min_size, :min_size].toarray())

        latent_parameters_marginal_sd = np.sqrt(Q_inverse_selected.diagonal())
        print_mpi(
            f"standard deviation fixed effects: {latent_parameters_marginal_sd[-self.pyinla_config.model.n_fixed_effects:]}"
        )

        return Q_inverse_selected
