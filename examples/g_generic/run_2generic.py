import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import GenericSubModel, RegressionSubModel
from dalia.utils import (
    extract_diagonal,
    print_msg,
    plot_marginal_distributions_hp,
    plot_prior_hp,
    save_to_json,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/inputs_2generic"

if __name__ == "__main__":
    print_msg("--- Example: Gaussian likelihood with 2 Generic submodels and some fixed effects ---")

    save_dalia_results = False  # Set to True to save results to JSON
    
    # Check for parsed parameters
    args = parse_args()
    
        # Configurations of the generic submodel
    generic_iid_dict = {
        "type": "generic",
        "input_dir": f"{BASE_DIR}/inputs_generic_iid",
        "tau": 1.5,  # has to be positive
        "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 1e-1},
        #"ph_tau": {"type": "halfnormal", "mean": 1.0, "precision": 0.5},
    }
    generic_iid = GenericSubModel(
        config=submodels_config.parse_config(generic_iid_dict),
    )

    # Configurations of the generic submodel
    generic_dict = {
        "type": "generic",
        "input_dir": f"{BASE_DIR}/inputs_generic_dense",
        "tau": 2.5,  # has to be positive
        "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 1e-1},
        #"ph_tau": {"type": "halfnormal", "mean": 1.0, "precision": 0.5},
    }
    generic_dense = GenericSubModel(
        config=submodels_config.parse_config(generic_dict),
    )
    
    # fixed effects submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_fixed_effects",
        "n_fixed_effects": 4,
    }
    
    fixed_effects = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )
    
    # Likelihood
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 10.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 5e-2},
    }
    # Creation of the first model by combining the Generic submodel and the likelihood
    model = Model(
        submodels=[generic_iid, generic_dense, fixed_effects],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)
    
    ## Plot prior of hyperparameter -- identification by [0], [1], ... not amazing but works for now
    # theta_interval = [1e-6, 150]
    # prior_hp = model.prior_hyperparameters[2]
    # fig, ax = plot_prior_hp("prec_o", theta_interval, prior_hp)
    # import matplotlib.pyplot as plt
    # plt.show()

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    theta_ref = xp.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")

    results = dalia.run()

    print_msg("\n--- Results ---")
    print_msg("theta reference:\n", theta_ref)
    print_msg("Theta values external:\n", results["theta"])
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    theta_ref = xp.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    print_msg("Reference theta:", theta_ref)
    print_msg(
        "Norm (theta - theta_ref) / Norm (theta_ref): ",
        f"{xp.linalg.norm(results['theta'] - theta_ref) / xp.linalg.norm(theta_ref):.4e}",
    )

    # Compare latent parameters
    x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")
    print_msg(
        "Norm (x - x_ref) / Norm (x_ref): ",
        f"{xp.linalg.norm(results['x'] - x_ref) / xp.linalg.norm(x_ref):.4e}",
    )
    
    ## print fixed effects estimates and reference
    x_ref_fe = x_ref[-model.n_fixed_effects:]  # last 4 elements are fixed effects
    x_est_fe = results["x"][-model.n_fixed_effects:]
    print_msg("Reference fixed effects:", x_ref_fe)
    print_msg("Estimated fixed effects:", x_est_fe)

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

    print_msg("\n--- Marginal distributions of the hyperparameters ---")
    marginals_hp = dalia.marginal_distributions_hp()

    fig, axes = plot_marginal_distributions_hp(marginals_hp)
    import matplotlib.pyplot as plt

    plt.savefig(f"gr_marginal_distributions_hp.png")
    
    tau = marginals_hp["hyperparameters"]["tau"]
    tau_2 = marginals_hp["hyperparameters"]["tau_2"]
    prec_obs = marginals_hp["hyperparameters"]["prec_o"]
    
    tau_quantile_pairs = tau["quantiles"]["external"]["pairs"]
    tau_2_quantile_pairs = tau_2["quantiles"]["external"]["pairs"]
    prec_quantile_pairs = prec_obs["quantiles"]["external"]["pairs"]

    print("Quantile pairs of tau:")
    for p, q in tau_quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")
        
    print("Quantile pairs of tau_2:")
    for p, q in tau_2_quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print("Quantile pairs of prec_o:")
    for p, q in prec_quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")
        
    # save estimates to reference outputs folder
    if save_dalia_results:
        save_to_json(
            results=results,
            filename=f"{BASE_DIR}/reference_outputs/dalia_estimates.json",
        )

    print_msg("\n--- Finished ---")
