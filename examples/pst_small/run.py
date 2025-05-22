import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np

from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from pyinla.utils import print_msg, get_host
from examples_utils.parser_utils import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Poisson Spatio-temporal model with regression ---")

    # Check for parsed parameters
    args = parse_args()

    # Configurations of the submodels
    # . Spatio-temporal submodel
    spatio_temporal_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0,
        "r_t": 0,
        "sigma_st": 2,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": -2.30258509299405, "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": 0.693147180559945, "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 1.38629436111989, "precision": 0.5},
    }
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_dict),
    )
    # . Regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_regression",
        "n_fixed_effects": 8,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    # Configurations of the likelihood
    likelihood_dict = {
        "type": "poisson",
        "input_dir": f"{BASE_DIR}",
    }

    # Creation of the model by combining the submodels and the likelihood
    model = Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    # Configurations of PyINLA
    pyinla_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-3,
            "disp": True,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    pyinla = PyINLA(
        model=model,
        config=pyinla_config.parse_config(pyinla_dict),
    )

    minimization_result = pyinla.minimize()

    print_msg("\n--- Results ---")
    print_msg("Theta values:\n", minimization_result["theta"])
    print_msg(
        "Mean of the fixed effects:\n",
        minimization_result["x"][-model.submodels[-1].n_fixed_effects :],
    )

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    theta_ref = np.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{np.linalg.norm(minimization_result['theta'] - get_host(theta_ref)):.4e}",
    )

    # Compare latent parameters
    x_ref = np.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(minimization_result['x'] - x_ref):.4e}",
    )

    print_msg("\n--- Finished ---")
