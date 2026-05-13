import os
import sys

import numpy as np

from pathlib import Path


from dalia import xp, sp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import GenericSubModel, RegressionSubModel
from dalia.utils import (
    extract_diagonal,
    print_msg,
    plot_marginal_distributions_hp,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(__file__).resolve().parent / "synthetic_data"


if __name__ == "__main__":
    print_msg("--- Example: Bee Genomics  ---")
    # consists of 2 generic submodels (iid + dense component) and 1 regression submodel (covariates + covariates) with a Gaussian likelihood

    # Check for parsed parameters
    args = parse_args()

    # Configurations of the generic submodel
    generic_dict_iid = {
        "type": "generic",
        "input_dir": f"{BASE_DIR}/inputs_iid",
        "tau": 4,  # has to be positive
        "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 5 * 1e-5},
    }
    generic_iid = GenericSubModel(
        config=submodels_config.parse_config(generic_dict_iid),
    )

    # Configurations of the generic submodel
    generic_dict_queenGRMinv = {
        "type": "generic",
        "input_dir": f"{BASE_DIR}/inputs_queenGRMinv",
        "tau": 4,  # has to be positive
        "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 5 * 1e-5},
    }
    generic_queenGRMinv = GenericSubModel(
        config=submodels_config.parse_config(generic_dict_queenGRMinv),
    )

    # fixed effects + intercept
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_fixed_effects",
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    # Likelihood
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 1.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 5 * 1e-5},
    }
    # Creation of the first model by combining the Generic submodel and the likelihood
    model = Model(
        submodels=[generic_iid, generic_queenGRMinv, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": 50,
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

    results = dalia.run()

    # load theta reference and set theta_internal to reference values
    theta_ref_internal = xp.load(f"{BASE_DIR}/reference_outputs/theta_internal.npy")
    x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x.npy")

    print_msg("\n--- Results ---")
    print_msg("theta reference internal:\n", theta_ref_internal)
    print_msg("Theta values external:\n", results["theta"])
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])
    # print_msg(
    #     "Mean of the latent parameters:\n",
    #     results["x"],
    # )

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    print_msg("Reference theta internal:", theta_ref_internal)
    print_msg(
        "Norm (theta internal - theta_ref_internal):        ",
        f"{xp.linalg.norm(results['theta_internal'] - theta_ref_internal):.4e}",
    )

    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref)/Norm(x_ref):                ",
        f"{xp.linalg.norm(results['x'] - x_ref) / xp.linalg.norm(x_ref):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional)
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    print_msg("\n--- Marginal distributions of the hyperparameters ---")
    marginals_hp = dalia.marginal_distributions_hp()

    prec_obs = marginals_hp["hyperparameters"]["prec_o"]
    quantile_pairs = prec_obs["quantiles"]["external"]["pairs"]

    print("Quantile pairs of prec_o:")
    for p, q in quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print_msg("\n--- Finished ---")
