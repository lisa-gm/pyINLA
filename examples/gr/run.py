import os

import numpy as np

from pyinla import xp
from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel
from pyinla.utils import extract_diagonal, get_host, print_msg

path = os.path.dirname(__file__)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    regression_dict = {
        "type": "regression",
        "input_dir": f"{base_dir}/inputs",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 1.5,
        "prior_hyperparameters": {"type": "gaussian", "mean": 3.5, "precision": 0.5},
    }
    model = Model(
        submodels=[regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )

    print_msg(model)

    print_msg("Submodules initialized.")
    pyinla_dict = {
        "solver": {"type": "dense"},
        "minimize": {"max_iter": 50, "gtol": 1e-3, "disp": True},
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    pyinla = PyINLA(
        model=model,
        config=pyinla_config.parse_config(pyinla_dict),
    )

    theta_ref = xp.load(f"{base_dir}/reference_outputs/theta_ref.npy")
    x_ref = xp.load(f"{base_dir}/reference_outputs/x_ref.npy")

    theta_ref = xp.array(theta_ref)
    x_ref = xp.array(x_ref)

    results = pyinla.run()
    
    print_msg("\n--- Results ---")
    print_msg("Theta values:\n", results["theta"])
    print_msg("Covariance of theta:\n", results["cov_theta"])
    print_msg(
        "Mean of the fixed effects:\n",
        results["x"][-model.submodels[-1].n_fixed_effects:],
    )

    print_msg("\n--- Comparisons ---")

    # Compare hyperparameters
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{np.linalg.norm(results['theta'] - get_host(theta_ref)):.4e}",
    )

    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(results['x'] - get_host(x_ref)):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = pyinla.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    # Compare marginal variances of observations
    var_obs = pyinla.get_marginal_variances_observations(theta=theta_ref, x_star=x_ref)
    var_obs_ref = extract_diagonal(model.a @ Qinv_ref @ model.a.T)
    print_msg(
        "Norm (var_obs - var_obs_ref):    ",
        f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    )

    print_msg("\n--- Finished ---")
