# Copyright 2024 pyINLA authors. All rights reserved.

import numpy as np
from scipy.optimize import minimize

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.models.regression import Regression
from pyinla.models.spatio_temporal import SpatioTemporal
from pyinla.prior_hyperparameters.gaussian import GaussianPriorHyperparameters
from pyinla.prior_hyperparameters.penalized_complexity import (
    PenalizedComplexityPriorHyperparameters,
)


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

        self.theta = None

        self.theta_initial = np.ones_like(self.theta)

    def run(self) -> None:
        """Fit the model using INLA."""

        self.theta_star = minimize(
            self._evaluate_f,
            self.theta_initial,
            method="BFGS",
        )

    def _evaluate_f(self):
        pass
