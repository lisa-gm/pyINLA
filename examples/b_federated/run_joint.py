import os
import sys

import numpy as np
import pandas as pd

from dalia import xp
from dalia.configs import (
    dalia_config,
    likelihood_config,
    submodels_config,
)
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import RegressionSubModel, GenericSubModel
from dalia.utils import print_msg

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Federated Binomial Model ---")

    # Check for parsed parameters
    args = parse_args()

    # random site-specific intercept
    random_intercept = False  # True

    data_type = "trauma"
    family = "binomial"

    if random_intercept:
        n_fixed_effects = 4  ## no global intercept, only covariates
        joint_folder = f"joint_{data_type}_{family}_site_specific_intercept"
    else:
        n_fixed_effects = 5  ## this includes global intercept
        joint_folder = f"joint_{data_type}_{family}_global_intercept"

    # Configurations of the regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/{joint_folder}/inputs_regression",
        "n_fixed_effects": n_fixed_effects,
        "fixed_effects_prior_precision": 0.1,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    # Likelihood
    likelihood_dict = {
        "type": "binomial",
        "input_dir": f"{BASE_DIR}/{joint_folder}",
    }

    if random_intercept:
        # setup generic submodel for random intercept
        generic_dict = {
            "type": "generic",
            "input_dir": f"{BASE_DIR}/{joint_folder}/inputs_generic",
            "tau": 4,  # has to be positive
            "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 1e-5},
        }
        generic = GenericSubModel(
            config=submodels_config.parse_config(generic_dict),
        )

        model = Model(
            submodels=[generic, regression],
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
        )
    else:
        # Creation of the first model by combining the Regression submodel and the likelihood
        model = Model(
            submodels=[regression],
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
        )

    print_msg(model)

    # Configurations of DALIA
    dalia_dict = {
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
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    # theta_ref = xp.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    # x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")

    results = dalia.run()

    print_msg("\n--- Results ---")
    fixed_effects_mean = results["x"][-model.n_fixed_effects :]
    print_msg(
        "Mean of the fixed effects:\n",
        fixed_effects_mean,
    )
    if random_intercept:
        print_msg(
            "Mean of the random intercepts:\n",
            results["x"][: model.submodels[0].n_latent_parameters :],
        )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    marginals_hp = dalia.marginal_distributions_hp()

    if random_intercept:
        n_random = model.submodels[0].n_latent_parameters
        random_effects_mean = results["x"][:n_random]
        random_sd = np.sqrt(var_latent_params[:n_random])
        random_ci_lower = random_effects_mean - 1.96 * random_sd
        random_ci_upper = random_effects_mean + 1.96 * random_sd

        fixed_sd = np.sqrt(
            var_latent_params[n_random : n_random + model.n_fixed_effects]
        )
        fixed_covariates = ["sex", "age", "ISS", "GCS"]
        random_covariates = [f"site_intercept_{idx}" for idx in range(1, n_random + 1)]

        tau_idx = list(model.theta_keys).index("tau")
        tau_estimate = float(results["theta"][tau_idx])
        tau_quantile_pairs = marginals_hp["hyperparameters"]["tau"]["quantiles"][
            "external"
        ]["pairs"]
        tau_ci_lower = float(tau_quantile_pairs[0][1])
        tau_ci_upper = float(tau_quantile_pairs[-1][1])
        tau_rows = [
            {
                "Estimate": tau_estimate,
                "lower": tau_ci_lower,
                "upper": tau_ci_upper,
                "Method": "DALIA",
                "Covariate": "precision_random_intercept",
            }
        ]
    else:
        fixed_sd = np.sqrt(var_latent_params[: model.n_fixed_effects])
        fixed_covariates = ["(Intercept)", "sex", "age", "ISS", "GCS"]
        random_effects_mean = np.array([])
        random_ci_lower = np.array([])
        random_ci_upper = np.array([])
        random_covariates = []
        tau_rows = []

    fixed_ci_lower = fixed_effects_mean - 1.96 * fixed_sd
    fixed_ci_upper = fixed_effects_mean + 1.96 * fixed_sd

    summary_rows = []

    for covariate, lower, upper, estimate in zip(
        fixed_covariates, fixed_ci_lower, fixed_ci_upper, fixed_effects_mean
    ):
        summary_rows.append(
            {
                "Estimate": estimate,
                "lower": lower,
                "upper": upper,
                "Method": "DALIA",
                "Covariate": covariate,
            }
        )

    for covariate, lower, upper, estimate in zip(
        random_covariates, random_ci_lower, random_ci_upper, random_effects_mean
    ):
        summary_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "Estimate": estimate,
                "Method": "DALIA",
                "Covariate": covariate,
            }
        )

    summary_rows.extend(tau_rows)

    df_dalia = pd.DataFrame(summary_rows)
    df_dalia.to_csv(f"{BASE_DIR}/dalia_summary_{data_type}_joint.csv", index=False)
    print_msg(df_dalia)

    print_msg("\n--- Finished ---")
