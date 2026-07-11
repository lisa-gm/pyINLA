from pathlib import Path

import numpy as np
from hyperparameter import (
    Hyperparameter,
    HyperparameterManager,
    HyperparameterManagerConfig,
)
from scipy.optimize import OptimizeResult, minimize

from . import inla
from .model import GenomicModel, GenomicModelConfig, StatisticalModel


def optimize(
    model : StatisticalModel,
    objective_function : callable,
    jacobian_function : callable,
    ) -> OptimizeResult:
    # Maybe the hyperparameter manager should be part of the optimization itself?
    # . This woudl imply that after optimizing a Model() its hyperparameters get updated and
    # these hyperparameters are then the one that are gonna be used for the "post-processing"
    # related computations.

    # Initialize the hyperparameter manager with the hyperparameters and their initial values
    hp_manager_config : HyperparameterManagerConfig = ...
    hp_manager : HyperparameterManager = HyperparameterManager(
        hyperparameters : List[Hyperparameter] = [tau_iid, tau_queen, prec_regression],
        config=hp_manager_config
    )

    # Perform the optimization of the hyperparameters using the objective function and jacobian.
    # . could be interesting to have a checkpointing function (save the current state of the optimization to disk) to allow for resuming the optimization in case of interruptions.
    initial_hyperparameters : np.ndarray = hp_manager.get_initial_hyperparameter_values()
    bounds : List[Tuple[float, float]] = hp_manager.get_hyperparameter_bounds()

    result : OptimizeResult = minimize(
        fun=objective_function,
        x0=initial_hyperparameters,
        jac=jacobian_function,
        bounds=bounds,
        method='L-BFGS-B'
    )

    return result

if __name__ == "__main__":
    # Configure and initialize the Genomic Model
    config : GenomicModelConfig = GenomicModelConfig(
        dataset_path=Path("path/to/dataset"),
        n_observations=100,
        # Component: iid
        iid_prior_n=100,
        iid_design_name="iid_design_matrix.npy",
        # Component: queen
        queen_prior_name="queen_matrix.npy",
        queen_design_name="queen_design_matrix.npy",
        # Component: regression
        regression_prior_n=10,
        regression_design_name="regression_design_matrix.npy"
    )

    model : GenomicModel = GenomicModel(config=config)
    
    # Optimize the model's hyperparameters using the defined objective function and jacobian.
    result : OptimizeResult = optimize(
        model=model,
        objective_function=inla.objective,
        jacobian_function=inla.jacobian,
    )
