# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from scipy.optimize import minimize
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

        # --- Load observation vector
        self.y = np.load(pyinla_config.input_dir / "y.npy")
        self.n_observations = self.y.shape[0]

        # --- Load design matrix
        self.a = load_npz(pyinla_config.input_dir / "a.npz")

        # --- Load latent parameters vector
        try:
            self.x = np.load(pyinla_config.input_dir / "x.npy")
        except FileNotFoundError:
            self.x = np.zeros((self.a.shape[1]), dtype=self.y.dtype)

        self._check_dimensions()

        # --- Initialize model
        if self.pyinla_config.model == "regression":
            self.model = Regression(pyinla_config)
        elif self.pyinla_config.model == "spatio-temporal":
            self.model = SpatioTemporal(pyinla_config)

        # --- Initialize prior hyperparameters
        if self.pyinla_config.prior_hyperparameters == "gaussian":
            self.prior_hyperparameters = GaussianPriorHyperparameters(pyinla_config)
        elif self.pyinla_config.prior_hyperparameters == "penalized_complexity":
            self.prior_hyperparameters = PenalizedComplexityPriorHyperparameters(
                pyinla_config
            )

        # --- Initialize likelihood
        if self.pyinla_config.likelihood.type == "gaussian":
            self.likelihood = GaussianLikelihood(pyinla_config, self.n_observations)
        elif self.pyinla_config.likelihood.type == "poisson":
            self.likelihood = PoissonLikelihood(pyinla_config, self.n_observations)
        elif self.pyinla_config.likelihood.type == "binomial":
            self.likelihood = BinomialLikelihood(pyinla_config, self.n_observations)

        # --- Initialize theta
        self.theta_initial = theta_dict2array(
            self.model.get_theta(), self.likelihood.get_theta()
        )

    def run(self) -> np.ndarray:
        """Fit the model using INLA."""

        result = minimize(
            self.theta_initial,
            self._evaluate_f,
            self._evaluate_grad_f,
            method="BFGS",
        )

        if result.success:
            print(
                "Optimization converged successfully after", result.nit, "iterations."
            )
            self.theta_star = result.x
            return True
        else:
            print("Optimization did not converge.")
            return False

    def get_theta_star(self) -> dict:
        """Get the optimal theta."""
        return theta_array2dict(
            self.theta_star, self.model.get_theta(), self.likelihood.get_theta()
        )

    def _check_dimensions_observations(self) -> None:
        """Check the dimensions of the model."""
        assert self.y.shape[0] == self.a.shape[0], "Dimensions of y and A do not match."
        assert self.x.shape[0] == self.a.shape[1], "Dimensions of x and A do not match."

    def _evaluate_f(self, theta: np.ndarray) -> float:
        theta_model, theta_likelihood = theta_array2dict(
            theta, self.model.get_theta(), self.likelihood.get_theta()
        )

        log_prior = self.prior_hyperparameters.evaluate_log_prior(
            theta_model, theta_likelihood
        )

        x = ...

        likelihood = self.likelihood.evaluate_likelihood(theta_likelihood, x)

        prior_latent_parameters = 0.0

        conditional_latent_parameters = 0.0

        return (
            log_prior
            + likelihood
            + prior_latent_parameters
            - conditional_latent_parameters
        )

    def _evaluate_grad_f(self):
        pass

    def _inner_iteration(self):
        pass
