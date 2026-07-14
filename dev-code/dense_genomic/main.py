from pathlib import Path

import numpy as np
from hyperparameter import (
    Hyperparameter,
    HyperparameterManager,
    HyperparameterManagerConfig,
    assemble_hyperparameter_dict,
)
from scipy.optimize import OptimizeResult, minimize

from . import inla
from .model import GenomicModel, GenomicModelConfig, StatisticalModel


def fit_model(
    model: StatisticalModel,
    options: dict = None,
) -> OptimizeResult:
    # Initialize the hyperparameter manager HPM
    hpm_config: HyperparameterManagerConfig = HyperparameterManagerConfig(
        # HPM State Checkpointing
        checkpoint_hpm=False,
        checkpoint_hpm_every=10,
        checkpoint_hpm_path=Path("hpm_checkpoint.pkl"),
        # Hyperparameter Optimization History Checkpointing
        track_history=False,
        checkpoint_history_every=10,
        checkpoint_history_path=Path("hpm_history_checkpoint.pkl"),
    )
    hpm: HyperparameterManager = HyperparameterManager(model=model, config=hpm_config)

    def minimize_callback(xk: np.ndarray):
        """Callback function for the optimization process.

        After each successful iteration of the optimization algorithm,
        this function will commit the buffer of tentative hyperparameter
        values in the HyperparameterManager's and flush it.

        Parameters
        ----------
        xk : np.ndarray
            The current hyperparameter values at the current iteration.
        """
        hpm.commit_buffer()

    # Fit the model's hyperparameters to the observations using the INLA objective function.
    model_fitting_result: OptimizeResult = minimize(
        # We always restart with the latest accepted hyperparameters
        # (if new optimization: initial values, if restarded
        # optimization: latest accepted values)
        x0=hpm.get_latest_hyperparameters(format="array", include_fixed=False),
        fun=inla.objective,
        jac=True,  # Assumes inla.objective returns (fun, jac)
        args=(model, hpm),
        bounds=hpm.get_bounds(),
        method="L-BFGS-B",
        callback=minimize_callback,
        options=options,
    )

    # Overwrite the model's hyperparameters value with the one
    # found by the optimization (latest accepted hyperparameters)
    hpm.update_model(model=model)

    return model_fitting_result


if __name__ == "__main__":
    # Configure and initialize the Genomic Model
    # . Configure the hyperparameters
    tau_iid: Hyperparameter = Hyperparameter(
        name="tau_iid",
        value=1.0,
    )
    tau_queen: Hyperparameter = Hyperparameter(
        name="tau_queen",
        value=1.0,
    )
    prec_regression: Hyperparameter = Hyperparameter(
        name="prec_regression",
        value=0.1,
        is_fixed=True,  # This hyperparameter is fixed and will not be optimized
    )
    genomic_hps: dict[str, Hyperparameter] = assemble_hyperparameter_dict(
        hyperparameters=[tau_iid, tau_queen, prec_regression]
    )

    # . Configure the Genomic Model
    config: GenomicModelConfig = GenomicModelConfig(
        path_to_model_components=Path("path/to/dataset"),
        path_to_observations=Path("path/to/observations.npy"),
        hyperparameters=genomic_hps,
        # Component: iid
        iid_prior_n=100,
        iid_design_name="iid_design_matrix.npy",
        # Component: queen
        queen_prior_name="queen_matrix.npy",
        queen_design_name="queen_design_matrix.npy",
        # Component: regression
        regression_prior_n=10,
        regression_design_name="regression_design_matrix.npy",
    )

    # Instanciate the Genomic Model
    model: GenomicModel = GenomicModel(config=config)

    # Optimize the model's hyperparameters using the defined objective function and jacobian.
    # . minimization options
    options = {
        "maxiter": 100,
        "maxcor": 10,
        "maxls": 100,
        "ftol": 1e-9,
        "gtol": 1e-5,
        "disp": False,
    }
    # . run model fitting
    model_fitting_result: OptimizeResult = fit_model(
        model=model,
        options=None,
    )
