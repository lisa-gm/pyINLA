import os
import sys

from dalia.configs import (
    dalia_config,
    likelihood_config,
    models_config,
    submodels_config,
)
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.models.federated_model import FederatedModel
from dalia.submodels import GenericSubModel, RegressionSubModel
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
    random_intercept = True

    data_type = "trauma"
    family = "binomial"
    if random_intercept:
        n_fixed_effects = 4  # no global intercept, only covariates
        split_folder = f"split_{data_type}_{family}_site_specific_intercept"
    else:
        n_fixed_effects = 5  # includes global intercept
        split_folder = f"split_{data_type}_{family}_global_intercept"

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
            "n_fixed_effects": n_fixed_effects,
            "fixed_effects_prior_precision": 0.1,
        }
        regression = RegressionSubModel(
            config=submodels_config.parse_config(regression_dict),
        )

        submodels = [regression]
        if random_intercept:
            generic_dict = {
                "type": "generic",
                "input_dir": f"{hospital_dir}/inputs_generic",
                "tau": 4,
                "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 1e-5},
            }
            generic = GenericSubModel(
                config=submodels_config.parse_config(generic_dict),
            )
            submodels = [generic, regression]

        likelihood_dict = {
            "type": "binomial",
            "input_dir": hospital_dir,
        }
        model_local = Model(
            submodels=submodels,
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

    results = dalia.run()

    print_msg("\n--- Results ---")
    fixed_effects_mean = results["x"][-federated_model.n_fixed_effects :]
    print_msg("Mean of the fixed effects:\n", fixed_effects_mean)

    if random_intercept:
        n_random = models[0].submodels[0].n_latent_parameters
        print_msg("Mean of the random intercepts:\n", results["x"][:n_random])

    print_msg("\n--- Finished ---")
