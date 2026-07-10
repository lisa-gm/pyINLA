import numpy as np

from scipy.optimize import minimize, OptimizeResult

from pathlib import Path

from hyperparameter import Hyperparameter, HyperparameterManagerConfig, HyperparameterManager
from .model import GenomicModel, GenomicModelConfig


def objective():
    # Define the objective function for the INLA optimization.
    # . need to caracterize the architectural difference differences between getting forward difference gradient (and objective function at current hp) and auto-differentiation.
    # . . in particulat objective(), jacobian(), and how they are plugged intot he optimize()
    
    # Conditional of the latent field
    # . In the Gaussian case:
    # . . Q_cond = Q_prior - theta_likelihood * a.T @ a
    
    conditional_latent_parameters : float = ...

    prior_latent_parameters : float = ...
    likelihood : float = ...
    prior_hyperparameters : float = ...

    f = (
        conditional_latent_parameters
        - prior_latent_parameters
        - likelihood
        - prior_hyperparameters
    )
    
    return f

def jacobian():
    ...
    # return grad separatly? Need to check scipy.minimize documentation for this.


def optimize(
    objective_function : callable,
    jacobian_function : callable,
    hyperparameter_manager : HyperparameterManager
    ) -> OptimizeResult:
    # Perform the optimization of the hyperparameters using the objective function and jacobian.
    # . could be interesting to have a checkpointing function (save the current state of the optimization to disk) to allow for resuming the optimization in case of interruptions.
    initial_hyperparameters : np.ndarray = hyperparameter_manager.get_initial_hyperparameter_values()
    bounds : List[Tuple[float, float]] = hyperparameter_manager.get_hyperparameter_bounds()

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
    
    # Initialize the hyperparameter manager with the hyperparameters and their initial values
    hp_manager_config : HyperparameterManagerConfig = ...
    hp_manager : HyperparameterManager = HyperparameterManager(
        hyperparameters : List[Hyperparameter] = [tau_iid, tau_queen, prec_regression],
        config=hp_manager_config
    )

    # Perform the hyperparameter optimization using the objective function and jacobian
    result : OptimizeResult = optimize(
        objective_function=objective,
        jacobian_function=jacobian,
        hyperparameter_manager=hp_manager
    )
