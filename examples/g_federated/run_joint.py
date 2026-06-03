import os
import sys

import numpy as np
import pandas as pd

from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import GenericSubModel, RegressionSubModel
from dalia.utils import (
    print_msg,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Gaussian Regression ---")

    # Check for parsed parameters
    args = parse_args()

    # random site-specific intercept
    random_intercept = False  # True

    data_type = "nurses_hom"
    family = "gaussian"
    # covariates excluding intercept
    # covariate_names = ["gender", "age", "experience", "wardtype"]
    covariate_names = ["age"]

    if random_intercept:
        n_fixed_effects = len(covariate_names)  # no global intercept, only covariates
        joint_folder = f"joint_{data_type}_{family}_site_specific_intercept"
        intercept_tag = "site_specific_intercept"
    else:
        n_fixed_effects = 1 + len(covariate_names)  # includes global intercept
        joint_folder = f"joint_{data_type}_{family}_global_intercept"
        intercept_tag = "global_intercept"

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
    ### prior not exactly matching yet?!
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 1.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 1e-5},
        # "prior_hyperparameters": {"type": "gaussian", "mean": 1.0, "precision": 0.5},
    }

    if random_intercept:
        generic_dict = {
            "type": "generic",
            "input_dir": f"{BASE_DIR}/{joint_folder}/inputs_generic",
            "tau": 4,
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
            "max_iter": 100,
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
    print_msg("Theta values external:\n", results["theta"])
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])
    fixed_effects_mean = results["x"][-model.n_fixed_effects :]
    print_msg(
        "Mean of the fixed effects:\n",
        fixed_effects_mean,
    )
    if random_intercept:
        n_random = model.submodels[0].n_latent_parameters
        random_effects_mean = results["x"][:n_random]
        print_msg("Mean of the random intercepts:\n", random_effects_mean)

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    fixed_effects_var = var_latent_params[-model.n_fixed_effects :]
    fixed_effects_sd = np.sqrt(fixed_effects_var)
    fixed_ci_lower = fixed_effects_mean - 1.96 * fixed_effects_sd
    fixed_ci_upper = fixed_effects_mean + 1.96 * fixed_effects_sd

    marginals_hp = dalia.marginal_distributions_hp()
    prec_obs = marginals_hp["hyperparameters"]["prec_o"]
    quantile_pairs = prec_obs["quantiles"]["external"]["pairs"]

    ## convert to sigma2 values
    sigma2_quantile_pairs = sorted(
        ((1.0 - p, 1.0 / q) for p, q in quantile_pairs),
        key=lambda pair: pair[0],
    )

    # store parameters in matching format as needed in R
    # make dataframe with columns:
    # - lower (2.5% quantile)
    # - upper (97.5% quantile)
    # - mean (mean of the fixed effect)
    # - Method (DALIA)
    # - covariate (intercept), gender, age, experience, wardtype, sigma2
    sigma2_lower = sigma2_quantile_pairs[0][1]
    sigma2_upper = sigma2_quantile_pairs[-1][1]
    prec_o_idx = list(model.theta_keys).index("prec_o")
    sigma2_mean = 1.0 / float(results["theta"][prec_o_idx])

    if random_intercept:
        random_sd = np.sqrt(var_latent_params[:n_random])
        random_ci_lower = random_effects_mean - 1.96 * random_sd
        random_ci_upper = random_effects_mean + 1.96 * random_sd
        fixed_covariates = covariate_names
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
                "lower": tau_ci_lower,
                "upper": tau_ci_upper,
                "Estimate": tau_estimate,
                "Method": "DALIA",
                "Covariate": "tau",
            }
        ]
    else:
        random_effects_mean = np.array([])
        random_ci_lower = np.array([])
        random_ci_upper = np.array([])
        fixed_covariates = ["(Intercept)"] + covariate_names
        random_covariates = []
        tau_rows = []

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
                "Estimate": estimate,
                "lower": lower,
                "upper": upper,
                "Method": "DALIA",
                "Covariate": covariate,
            }
        )

    summary_rows.extend(tau_rows)
    summary_rows.append(
        {
            "Estimate": sigma2_mean,
            "lower": sigma2_lower,
            "upper": sigma2_upper,
            "Method": "DALIA",
            "Covariate": "sigma2",
        }
    )

    df_dalia = pd.DataFrame(summary_rows)
    df_dalia.to_csv(
        f"{BASE_DIR}/dalia_summary_{data_type}_joint_{intercept_tag}.csv",
        index=False,
    )
    print_msg(df_dalia)

    print_msg("\n--- Finished ---")
