import os
import sys

import numpy as np

from dalia import xp
from dalia.configs import (
    dalia_config,
    likelihood_config,
    submodels_config,
    models_config,
)
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.models import ReplicateModel
from dalia.submodels import AR1SubModel, RegressionSubModel
from dalia.utils import (
    print_msg,
    save_to_json,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: AR1 with Regression and multiple replicates ---")

    save_dalia_results = True  # Set to True to save results to JSON
    
    # Check for parsed parameters
    args = parse_args()

    n_replicates = 1  # number of replicates

    # setup 1 model for each replicate
    models = []
    for i in range(n_replicates):
        # AR1 submodel
        ar1_dict = {
            "type": "ar1",
            "input_dir": f"{BASE_DIR}/inputs/replicate_{i+1}/inputs_ar1",
            "phi": 0.5,  # has to be between 0 and 1
            "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
            # initial guess on the precision
            "tau": 3,  # has to be positive
            "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
        }
        ar1 = AR1SubModel(
            config=submodels_config.parse_config(ar1_dict),
        )

        # Regression submodel
        regression_dict = {
            "type": "regression",
            "input_dir": f"{BASE_DIR}/inputs/replicate_{i+1}/inputs_regression",
            "n_fixed_effects": 1,
            "fixed_effects_prior_precision": 0.001,
        }
        regression = RegressionSubModel(
            config=submodels_config.parse_config(regression_dict),
        )

        likelihood_dict = {
            "type": "gaussian",
            "prec_o": 4.0,
            "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 1e-1},
        }
        local_model = Model(
            submodels=[ar1, regression],
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
        )
        models.append(local_model)

    replicate_dict = {
        "type": "replicate",
        "n_replicates": len(models),
        "theta": models[0].theta_external.tolist(),
        "theta_keys": list(models[0].theta_keys),
    }
    replicate_model = ReplicateModel(
        models=models,
        replicate_model_config=models_config.parse_config(replicate_dict),
    )
    print_msg(replicate_model)

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=replicate_model,
        config=dalia_config.parse_config(dalia_dict),
    )

    theta_ref = xp.load(f"{BASE_DIR}/reference_outputs/theta_original.npy")
    x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")

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
        "Norm (x - x_ref)/ norm(x_ref):   ",
        f"{xp.sqrt(xp.sum((results['x'] - x_ref) ** 2)) / xp.sqrt(xp.sum(x_ref ** 2)):.4e}",
    )
    
    hess_internal = dalia.hess_theta_internal
    print_msg("Hessian of theta internal:\n", hess_internal)

    # Compare marginal variances of latent parameters
    # var_latent_params = results["marginal_variances_latent"]
    # Qconditional = dalia.model.construct_Q_conditional(
    #     eta=replicate_model.a @ replicate_model.x
    # )
    # Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    # print_msg(
    #     "Norm (marg var latent - ref):    ",
    #     f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    # )

    # # Compare marginal variances of observations
    # var_obs = dalia.get_marginal_variances_observations()

    # var_obs_ref = extract_diagonal(replicate_model.a @ Qinv_ref @ replicate_model.a.T)
    # print_msg(
    #     "Norm (var_obs - var_obs_ref):    ",
    #     f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    # )

    print_msg("\n--- Marginal distributions of the hyperparameters ---")
    marginals_hp = dalia.marginal_distributions_hp()

    # Extract all hyperparameters
    phi = marginals_hp["hyperparameters"]["phi"]
    tau = marginals_hp["hyperparameters"]["tau"]
    prec_o = marginals_hp["hyperparameters"]["prec_o"]

    print("Quantiles of phi:")
    phi_quantile_pairs = phi["quantiles"]["external"]["pairs"]
    for p, q in phi_quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print("Quantiles of tau:")
    tau_quantile_pairs = tau["quantiles"]["external"]["pairs"]
    for p, q in tau_quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print("Quantiles of prec_o:")
    prec_quantile_pairs = prec_o["quantiles"]["external"]["pairs"]
    for p, q in prec_quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    # save estimates to reference outputs folder
    # import json

    # dalia_estimates = {
    #     "theta_internal": results["theta_internal"].tolist(),
    #     "theta_external": results["theta"].tolist(),
    #     "x": results["x"].tolist(),
    #     "cov_theta_internal_diagonal": xp.diag(results["cov_theta_internal"]).tolist(),
    #     "cov_theta_internal_full": results["cov_theta_internal"].tolist(),
    #     "hyperparameters": {
    #         "phi": {
    #             "mean": phi["mean_external"],
    #             "variance": phi["variance_external"],
    #             "quantile_pairs": phi_quantile_pairs,
    #             "pdf_pairs": list(zip(phi["pdf_data"][0].tolist(), phi["pdf_data"][1].tolist())),
    #         },
    #         "tau": {
    #             "mean": tau["mean_external"],
    #             "variance": tau["variance_external"],
    #             "quantile_pairs": tau_quantile_pairs,
    #             "pdf_pairs": list(zip(tau["pdf_data"][0].tolist(), tau["pdf_data"][1].tolist())),
    #         },
    #         "prec_o": {
    #             "mean": prec_o["mean_external"],
    #             "variance": prec_o["variance_external"],
    #             "quantile_pairs": prec_quantile_pairs,
    #             "pdf_pairs": list(zip(prec_o["pdf_data"][0].tolist(), prec_o["pdf_data"][1].tolist())),
    #         },
    #     },
    # }

    # reference_outputs_dir = f"{BASE_DIR}/reference_outputs"
    # os.makedirs(reference_outputs_dir, exist_ok=True)
    # with open(f"{reference_outputs_dir}/dalia_estimates.json", "w") as f:
    #     json.dump(dalia_estimates, f, indent=2)

    # save estimates to reference outputs folder
    if save_dalia_results:
        save_to_json(
            results=results,
            filename=f"{BASE_DIR}/reference_outputs/dalia_estimates.json",
        )