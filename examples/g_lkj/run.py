import os
import sys

import numpy as np

from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import LKJSubModel
from dalia.utils import (
    extract_diagonal,
    print_msg,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: LKJ Submodel ---")

    # Check for parsed parameters
    args = parse_args()

    # Configurations of the LKJ submodel
    # 2D latent with LKJ prior on correlation and HalfNormal on variances
    lkj_dict = {
        "type": "lkj",
        "input_dir": f"{BASE_DIR}/inputs_lkj",
        # Initial guesses on hyperparameters (external space)
        "sigma1": 1.0,
        "sigma2": 1.0,
        "rho": 0.0,
        # Prior hyperparameters
        "ph_sigma1": {"type": "half_normal", "precision": 0.5},  # variance = 2
        "ph_sigma2": {"type": "half_normal", "precision": 0.5},  # variance = 2
        "ph_rho": {"type": "lkj_2d", "eta": 2.0},
    }
    lkj = LKJSubModel(
        config=submodels_config.parse_config(lkj_dict),
    )

    # Likelihood
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 1.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 1e-1},
    }
    # Creation of the model by combining the LKJ submodel and the likelihood
    model = Model(
        submodels=[lkj],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    print(f"Internal theta values: {model.theta_internal}")
    print(f"External theta values: {model.theta_external}")
    Qprior = model.construct_Q_prior()
    print(f"Q_prior:\n{Qprior.toarray()}")

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
    theta_est = results["theta"]
    print_msg("theta estimated:\n", theta_est)
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

    # print_msg("\n--- Marginal distributions of the hyperparameters ---")
    # marginals_hp = dalia.marginal_distributions_hp()

    # fig, axes = plot_marginal_distributions_hp(marginals_hp)
    # import matplotlib.pyplot as plt

    # plt.savefig(f"lkj_marginal_distributions_hp.png")

    ## construct estimated LKJ covariance matrix of latent parameters
    lkj_est = xp.array(
        [
            [
                theta_est[0] ** 2,
                theta_est[0] * theta_est[1] * theta_est[2],
            ],
            [
                theta_est[0] * theta_est[1] * theta_est[2],
                theta_est[1] ** 2,
            ],
        ]
    )

    lkj_ref = np.array(
        [
            [theta_ref[0] ** 2, theta_ref[0] * theta_ref[1] * theta_ref[2]],
            [theta_ref[0] * theta_ref[1] * theta_ref[2], theta_ref[1] ** 2],
        ]
    )
    print_msg("\n--- LKJ Covariance Matrix of Latent Parameters ---")
    print_msg("Estimated LKJ covariance matrix:\n", lkj_est)
    print_msg("Reference LKJ covariance matrix:\n", lkj_ref)

    print_msg("\n--- Finished ---")
