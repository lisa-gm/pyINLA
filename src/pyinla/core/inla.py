# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from scipy.optimize import minimize

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

        # Initialize model
        if self.pyinla_config.model == "regression":
            self.model = Regression(pyinla_config)
        elif self.pyinla_config.model == "spatio-temporal":
            self.model = SpatioTemporal(pyinla_config)

        # Initialize prior hyperparameters
        if self.pyinla_config.prior_hyperparameters == "gaussian":
            self.prior_hyperparameters = GaussianPriorHyperparameters(pyinla_config)
        elif self.pyinla_config.prior_hyperparameters == "penalized_complexity":
            self.prior_hyperparameters = PenalizedComplexityPriorHyperparameters(
                pyinla_config
            )

        # Initialize likelihood
        if self.pyinla_config.likelihood.type == "gaussian":
            self.likelihood = GaussianLikelihood(pyinla_config)
        elif self.pyinla_config.likelihood.type == "poisson":
            self.likelihood = PoissonLikelihood(pyinla_config)
        elif self.pyinla_config.likelihood.type == "binomial":
            self.likelihood = BinomialLikelihood(pyinla_config)

        # Initialize theta
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

    def _evaluate_f(self, theta: np.ndarray) -> float:
        theta_model, theta_likelihood = theta_array2dict(
            theta, self.model.get_theta(), self.likelihood.get_theta()
        )

        log_prior = self.prior_hyperparameters.evaluate_log_prior(
            theta_model, theta_likelihood
        )

        likelihood = self.likelihood.evaluate_likelihood(theta_likelihood)

        prior_lattent_parameters = ...

        conditional_lattent_parameters = ...

        return (
            log_prior
            + likelihood
            + prior_lattent_parameters
            + conditional_lattent_parameters
        )

    def _evaluate_grad_f(self):
        pass
