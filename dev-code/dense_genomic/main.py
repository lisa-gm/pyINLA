import numpy as np

from scipy.optimize import minimize, OptimizeResult

from pathlib import Path

from hyperparameter import Hyperparameter, HyperparameterManagerConfig, HyperparameterManager




def update_prior_precision_matrix(
    Q_prior : np.ndarray,
    n_iid : int,
    n_queen : int,
    n_regression : int,
    hp_manager : HyperparameterManager
    ) -> np.ndarray:
    # Update the prior precision matrix with the new hyperparameters.
    # the scaling factors are computed as the ratio of the new hyperparameter value to the previous one.
    tau_iid_scaling = hp_manager.get_hyperparameter_value("tau_iid") / hp_manager.get_previous_hyperparameter_value("tau_iid")
    tau_queen_scaling = hp_manager.get_hyperparameter_value("tau_queen") / hp_manager.get_previous_hyperparameter_value("tau_queen")
    prec_regression_scaling = hp_manager.get_hyperparameter_value("prec_regression") / hp_manager.get_previous_hyperparameter_value("prec_regression")

    Q_prior[:n_iid, :n_iid] *= tau_iid_scaling
    Q_prior[n_iid:n_iid+n_queen, n_iid:n_iid+n_queen] *= tau_queen_scaling
    Q_prior[n_iid+n_queen:, n_iid+n_queen:] *= prec_regression_scaling

    return Q_prior


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
    # Set-up of the problem
    dataset_path : Path = ...

    n_iid : int = ...
    n_queen : int = ...
    n_regression : int = ...
    n_total : int = n_iid + n_queen + n_regression

    Q_prior : np.ndarray = np.zeros((n_total, n_total), dtype=np.float64)

    # Assemble the sub-components of the prior precision matrix
    # . prior_iid
    tau_iid : Hyperparameter = ...
    Q_prior[:n_iid, :n_iid] = np.eye(n_iid) * tau_iid
    # . prior_queen
    tau_queen : Hyperparameter = ...
    Q_prior[n_iid:n_iid+n_queen, n_iid:n_iid+n_queen] = tau_queen * np.load(dataset_path / "Q_queen.npy")
    # . prior_regression
    prec_regression : Hyperparameter = ...
    Q_prior[n_iid+n_queen:, n_iid+n_queen:] = np.eye(n_regression) * prec_regression

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

    # Problems:
    # - how do I interface the precision matrices construction to the objective function
    # - need a Model() class again?