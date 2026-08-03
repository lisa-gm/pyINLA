import os
import sys

import numpy as np

from dalia import xp, sp
from dalia.configs import (
    dalia_config,
    likelihood_config,
    submodels_config,
)
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import LKJSubModel, RegressionSubModel
from dalia.utils import (
    extract_diagonal,
    print_msg,
    save_to_json,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: LKJ Submodel with Multiple Groups (Replicated) ---")

    save_dalia_results = True  # Set to True to save results to JSON

    # Check for parsed parameters
    args = parse_args()

    n_groups = 5  # Number of groups

    # LKJ submodel: replicated for each group
    # Each replicate handles [u_g, v_g] (group-specific random intercept and slope)
    # All replicates share the same hyperparameters (sigma1, sigma2, rho)
    lkj_dict = {
        "type": "lkj",
        "input_dir": f"{BASE_DIR}/inputs_ngroups{n_groups}/inputs_lkj",
        "n_replicates": n_groups,  # One LKJ submodel per group
        "replicate_a": False,  # design matrix is already replicated in input file
        # Initial guesses on hyperparameters (external space)
        "sigma1": 1.0,  # variance of random intercept
        "sigma2": 1.0,  # variance of random slope
        "rho": 0.5,  # correlation between random intercept and slope
        # Prior hyperparameters
        "ph_sigma1": {"type": "half_normal", "precision": 0.5},  # variance = 2
        "ph_sigma2": {"type": "half_normal", "precision": 0.5},  # variance = 2
        "ph_rho": {"type": "lkj_2d", "eta": 1.0},
    }
    lkj = LKJSubModel(
        config=submodels_config.parse_config(lkj_dict),
    )

    # Regression submodel: global fixed effects (not replicated)
    # Handles [ beta_0]
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/ngroups{n_groups}/inputs_regression",
        "n_replicates": 1,  # Single shared fixed effects
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    # Likelihood
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 1.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 1e-1},
    }

    # Creation of the model by combining the replicated LKJ and regression submodels
    model = Model(
        submodels=[lkj, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
        input_dir=f"{BASE_DIR}/ngroups{n_groups}",
    )
    print_msg(model)

    print_msg(f"Internal theta values: {model.theta_internal}")
    print_msg(f"External theta values: {model.theta_external}")
    Qprior = model.construct_Q_prior()
    print_msg(f"Q_prior shape: {Qprior.shape}")

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    theta_ref = xp.load(
        f"{BASE_DIR}/ngroups{n_groups}/reference_outputs/theta_ref.npy"
    )
    x_ref = xp.load(f"{BASE_DIR}/ngroups{n_groups}/reference_outputs/x_ref.npy")

    results = dalia.run()

    print_msg("\n--- Results ---")
    print_msg("Theta values external:\n", results["theta"])
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    print_msg("Reference theta:", theta_ref)
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{xp.linalg.norm(results['theta'] - theta_ref):.4e}",
    )

    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref):                ",
        f"{xp.linalg.norm(results['x'] - x_ref):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    # Compare marginal variances of observations
    var_obs = dalia.get_marginal_variances_observations(
        theta_external=theta_ref, x_star=x_ref
    )
    var_obs_ref = extract_diagonal(model.a @ Qinv_ref @ model.a.T)
    print_msg(
        "Norm (var_obs - var_obs_ref):    ",
        f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    )

    ## Construct estimated LKJ covariance matrix (shared across all groups)
    lkj_est = xp.array(
        [
            [
                results["theta"][0] ** 2,
                results["theta"][0] * results["theta"][1] * results["theta"][2],
            ],
            [
                results["theta"][0] * results["theta"][1] * results["theta"][2],
                results["theta"][1] ** 2,
            ],
        ]
    )

    lkj_ref = np.array(
        [
            [theta_ref[0] ** 2, theta_ref[0] * theta_ref[1] * theta_ref[2]],
            [theta_ref[0] * theta_ref[1] * theta_ref[2], theta_ref[1] ** 2],
        ]
    )
    print_msg("\n--- LKJ Covariance Matrix of Latent Parameters (Shared) ---")
    print_msg("Estimated LKJ covariance matrix:\n", lkj_est)
    print_msg("Reference LKJ covariance matrix:\n", lkj_ref)

    print_msg("\n--- Global Fixed Effects ---")
    global_idx_start = 2 * n_groups
    print_msg(
        f"Reference global effects [ beta_0]: {x_ref[global_idx_start:]}"
    )
    print_msg(
        f"Estimated global effects [ beta_0]: {results['x'][global_idx_start:]}"
    )

    print_msg("\n--- Group-specific Random Effects ---")
    for group_id in range(1, n_groups + 1):
        print_msg(f"Group {group_id} random effects: {results['x'][2*(group_id-1):2*group_id]}")
