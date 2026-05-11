import os
import sys

import numpy as np
import pandas as pd

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
from dalia.submodels import RegressionSubModel
from dalia.utils import print_msg

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Federated Binomial Model ---")

    # Check for parsed parameters
    args = parse_args()

    data_type = "trauma"
    family = "binomial"
    split_folder = f"split_{data_type}_{family}"

    split_root = os.path.join(BASE_DIR, split_folder)
    hospital_dirs = [
        os.path.join(split_root, "hospital_1"),
        os.path.join(split_root, "hospital_2"),
        os.path.join(split_root, "hospital_3"),
    ]

    models = []
    for hospital_dir in hospital_dirs:
        regression_dict = {
            "type": "regression",
            "input_dir": f"{hospital_dir}/inputs_regression",
            "n_fixed_effects": 5,
            "fixed_effects_prior_precision": 0.001,
        }
        regression = RegressionSubModel(
            config=submodels_config.parse_config(regression_dict),
        )

        likelihood_dict = {
            "type": "binomial",
            "input_dir": hospital_dir,
        }
        model_local = Model(
            submodels=[regression],
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
        )
        models.append(model_local)

    print_msg(f"Constructed {len(models)} local models from split data")

    # Federated setup from local models.
    federated_dict = {
        "type": "federated",
        "n_models": len(models),
        "theta": models[0].theta_external.tolist(),
        "theta_keys": list(models[0].theta_keys),
    }
    federated_model = FederatedModel(
        models=models,
        federated_model_config=models_config.parse_config(federated_dict),
    )

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
        model=federated_model,
        config=dalia_config.parse_config(dalia_dict),
    )

    # theta_ref = xp.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    # x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")

    results = dalia.run()

    print_msg("\n--- Results ---")
    fixed_effects_mean = results["x"][-federated_model.n_fixed_effects :]
    print_msg(
        "Mean of the fixed effects:\n",
        fixed_effects_mean,
    )

    # # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    fixed_effects_var = var_latent_params[-federated_model.n_fixed_effects :]
    fixed_effects_sd = np.sqrt(fixed_effects_var)
    ci_lower = fixed_effects_mean - 1.96 * fixed_effects_sd
    ci_upper = fixed_effects_mean + 1.96 * fixed_effects_sd

    # store parameters in matching format as needed in R
    # make dataframe with columns:
    # - lower (2.5% quantile)
    # - upper (97.5% quantile)
    # - mean (mean of the fixed effect)
    # - Method (DALIA)
    # - covariate (intercept), gender, age, ISS, GCS

    df_dalia = pd.DataFrame(
        {
            "lower": ci_lower,
            "upper": ci_upper,
            "Estimate": fixed_effects_mean,
            "Method": "DALIA-FED",
            "Covariate": [
                "(Intercept)",
                "sex",
                "age",
                "ISS",
                "GCS",
            ],
        }
    )
    df_dalia.to_csv(f"{BASE_DIR}/dalia_summary_{data_type}.csv", index=False)

    print_msg(df_dalia)

    print_msg("\n--- Finished ---")
