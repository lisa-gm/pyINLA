# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC, abstractmethod

from pyinla.core.pyinla_config import PriorHyperparametersConfig

from pyinla.core.submodel import SubModel
from pyinla.submodels.regression import RegressionModel
from pyinla.submodels.spatio_temporal import SpatioTemporalModel

from pyinla.core.likelihood import Likelihood
from pyinla.likelihoods.binomial import BinomialLikelihood
from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.likelihoods.poisson import PoissonLikelihood


class PriorHyperparameters(ABC):
    """Abstract core class for prior hyperparameters."""

    def __init__(
        self,
        ph_config: PriorHyperparametersConfig,
        hyperparameter_type: str,
    ) -> None:
        """Initializes the prior hyperparameters."""

        self.ph_config = ph_config
        self.hyperparameter_type = hyperparameter_type

    @abstractmethod
    def evaluate_log_prior(self, theta: float) -> float:
        """Evaluate the log prior hyperparameters."""
        pass
