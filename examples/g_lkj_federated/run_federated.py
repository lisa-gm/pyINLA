import os
import sys

from dalia import xp
from dalia.configs import (
    dalia_config,
    likelihood_config,
    models_config,
    submodels_config,
)
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.models.federated_model import FederatedModel
from dalia.submodels import LKJSubModel, RegressionSubModel
from dalia.utils import (
    print_msg,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: LKJ Submodel with Multiple Groups (Federated Model) ---")

    # Check for parsed parameters
    args = parse_args()

    n_sites = 50  # Number of groups/sites
    print_msg(f"Using data from {n_sites} sites")

    # Construct a local model for each site
    models = []
    for site_id in range(1, n_sites + 1):
        site_dir = f"{BASE_DIR}/inputs_ngroups{n_sites}_federated/site{site_id}"
        
        # LKJ submodel for this site (2D: intercept and slope)
        lkj_dict = {
            "type": "lkj",
            "input_dir": f"{site_dir}/inputs_lkj",
            "n_replicates": 1,  # Single site, no replication within site
            # Initial guesses on hyperparameters (external space)
            "sigma1": 1.0,  # variance of random intercept
            "sigma2": 1.0,  # variance of random slope
            "rho": 0.5,  # correlation between random intercept and slope
            # Prior hyperparameters
            "ph_sigma1": {"type": "half_normal", "precision": 0.5},
            "ph_sigma2": {"type": "half_normal", "precision": 0.5},
            "ph_rho": {"type": "lkj_2d", "eta": 1.0},
        }
        lkj = LKJSubModel(
            config=submodels_config.parse_config(lkj_dict),
        )

        # Regression submodel for this site (global slope)
        regression_dict = {
            "type": "regression",
            "input_dir": f"{site_dir}/inputs_regression",
            "n_fixed_effects": 1,  # Global slope
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

        # Local model for this site
        model_local = Model(
            submodels=[lkj, regression],
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
            input_dir=site_dir,
        )
        models.append(model_local)

    print_msg(f"Constructed {len(models)} local models from {n_sites} sites")

    # Construct federated model from all local models
    # The key: all sites share the same hyperparameters (sigma1, sigma2, rho)
    federated_dict = {
        "type": "federated",
        "n_models": len(models),
        "theta": models[0].theta_external.tolist(),
        "theta_keys": list(models[0].theta_keys),
        # submodel order is [LKJSubModel, RegressionSubModel]
        # LKJ should be site-specific random, regression shared fixed
        "effect_type": ["random", "fixed"],
    }
    federated_model = FederatedModel(
        models=models,
        federated_model_config=models_config.parse_config(federated_dict),
    )

    print_msg(federated_model)

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=federated_model,
        config=dalia_config.parse_config(dalia_dict),
    )

    theta_ref = xp.load(
        f"{BASE_DIR}/inputs_ngroups{n_sites}_federated/reference_outputs/theta_ref.npy"
    )
    x_ref = xp.load(f"{BASE_DIR}/inputs_ngroups{n_sites}_federated/reference_outputs/x_ref.npy")

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

    ## Construct estimated LKJ covariance matrix (shared across all sites)
    lkj_est = lkj.lkj_covariance_matrix(results["theta"][:3])
    lkj_ref = lkj.lkj_covariance_matrix(theta_ref[:3])
    
    print_msg("\n--- LKJ Covariance Matrix of Latent Parameters (Shared Across Sites) ---")
    print_msg("Estimated LKJ covariance matrix:\n", lkj_est)
    print_msg("Reference LKJ covariance matrix:\n", lkj_ref)

    print_msg("\n--- Global Slope ---")
    print_msg(f"Reference global slope: {x_ref[-1]}")
    print_msg(f"Estimated global slope: {results['x'][-1]}")
    
    if n_sites <= 10:
        print_msg("\n--- Site-specific Random Effects ---")
        for site_id, site_idx in enumerate(range(1, n_sites + 1), start=1):
            print_msg(f"Site {site_idx} random effects: {results['x'][2*(site_idx-1):2*site_idx]}")

