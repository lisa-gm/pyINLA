# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy import sparse

from pyinla.core.likelihood import Likelihood
from pyinla.core.model import Model
from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.likelihoods.gaussian import GaussianLikelihood
from pyinla.models.regression import RegressionModel
from pyinla.models.spatio_temporal import SpatioTemporalModel
from pyinla.solvers.scipy_solver import ScipySolver
from pyinla.solvers.serinv_solver import SerinvSolverCPU
from pyinla.utils import sigmoid

from os import environ

environ["OMP_NUM_THREADS"] = "1"
