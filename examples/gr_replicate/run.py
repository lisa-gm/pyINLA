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
from dalia.submodels import RegressionSubModel
from dalia.utils import (
    extract_diagonal,
    print_msg,
    plot_marginal_distributions_hp,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Gaussian Regression with multiple replicates ---")

    # Check for parsed parameters
    args = parse_args()

    n_replicates = 20  # number of replicates

    # setup 1 model for each replicate
    models = []
    for i in range(n_replicates):
        # Configurations of the regression submodel
        path_dir = (
            f"{BASE_DIR}/inputs_nrep{n_replicates}/replicate_{i+1}/inputs_regression"
        )
        regression_dict = {
            "type": "regression",
            "input_dir": path_dir,
            "n_fixed_effects": 6,
        }
        regression = RegressionSubModel(
            config=submodels_config.parse_config(regression_dict),
        )
        likelihood_dict = {
            "type": "gaussian",
            "prec_o": 1.0,
            "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 1e-1},
        }
        local_model = Model(
            submodels=[regression],
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
        )
        models.append(local_model)

    print_msg(models)

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

    # Qprior = replicate_model.construct_Q_prior()
    # print(f"Q_prior:\n{Qprior.toarray()}")

    # Qconditional = replicate_model.construct_Q_conditional(
    #     eta=replicate_model.a @ replicate_model.x
    # )
    # print(f"Q_conditional:\n{Qconditional.toarray()}")

    # import matplotlib.pyplot as plt

    # plt.matshow(Qprior.toarray())
    # plt.show()

    # plt.matshow(Qconditional.toarray())
    # plt.show()

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=replicate_model,
        config=dalia_config.parse_config(dalia_dict),
    )

    theta_ref = xp.load(
        f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/theta_ref.npy"
    )
    x_ref = xp.load(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/x_ref.npy")

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
        "Norm (x - x_ref) / ||x_ref||:                ",
        f"{xp.sqrt(xp.sum((results['x'] - x_ref) ** 2)) / xp.sqrt(xp.sum(x_ref ** 2)):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(
        eta=replicate_model.a @ replicate_model.x
    )
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    # Compare marginal variances of observations
    var_obs = dalia.get_marginal_variances_observations()

    var_obs_ref = extract_diagonal(replicate_model.a @ Qinv_ref @ replicate_model.a.T)
    print_msg(
        "Norm (var_obs - var_obs_ref):    ",
        f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    )

    print_msg("\n--- Marginal distributions of the hyperparameters ---")
    marginals_hp = dalia.marginal_distributions_hp()

    fig, axes = plot_marginal_distributions_hp(marginals_hp)
    import matplotlib.pyplot as plt

    plt.savefig(f"gr_marginal_distributions_hp.png")

    prec_obs = marginals_hp["hyperparameters"]["prec_o"]
    quantile_pairs = prec_obs["quantiles"]["external"]["pairs"]
    pdf_pairs_x, pdf_pairs_y = prec_obs["pdf_data"]

    print("Quantile pairs of prec_o:")
    for p, q in quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    # save estimates to reference outputs folder
    import json

    dalia_estimates = {
        "theta_internal": results["theta_internal"].tolist(),
        "theta_external_map": results["theta"].tolist(),
        "theta_external_mean": prec_obs["mean_external"],
        "x": results["x"].tolist(),
        "cov_theta_internal_diagonal": xp.diag(results["cov_theta_internal"]).tolist(),
        "cov_theta_internal_full": results["cov_theta_internal"].tolist(),
        "marginal_variance_external_prec_o": prec_obs["variance_external"],
        "quantile_pairs": quantile_pairs,
        "pdf_pairs": list(zip(pdf_pairs_x.tolist(), pdf_pairs_y.tolist())),
    }
    
    reference_outputs_dir = f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs"
    os.makedirs(reference_outputs_dir, exist_ok=True)
    with open(f"{reference_outputs_dir}/dalia_estimates.json", "w") as f:
        json.dump(dalia_estimates, f, indent=2)

    print_msg("\n--- Finished ---")
