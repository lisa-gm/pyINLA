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

    # minimization_result = pyinla.minimize()

    # print_msg("\n--- PyINLA results ---")
    # print_msg("Final theta: ", minimization_result["theta"])
    # print_msg("Final f:", minimization_result["f"])
    # print_msg("final grad_f:", minimization_result["grad_f"])

    # print_msg("\n--- References ---")
    theta_ref = np.load(f"{base_dir}/reference_outputs/theta_ref.npy")
    x_ref = np.load(f"{base_dir}/reference_outputs/x_ref.npy")

    theta_ref = xp.array(theta_ref)
    x_ref = xp.array(x_ref)
    # print_msg("theta_ref: ", theta_ref)
    # print_msg(
    #     "Norm between thetas and theta_ref: ",
    #     np.linalg.norm(minimization_result["theta"] - theta_ref),
    # )
    # print_msg(
    #     "Norm between x and x_ref: ", np.linalg.norm(minimization_result["x"] - x_ref)
    # )

    # test hessian
    hess = pyinla._evaluate_hessian_f(theta_ref)
    print_msg("hessian: \n", hess)

    cov = pyinla.compute_covariance_hp(theta_ref)
    print_msg("covariance: \n", cov)

    # theta_ref = theta_ref + 1
    # compute marginal covariances latent parameters
    var_latent_params = pyinla.get_marginal_variances_latent_parameters(
        theta=theta_ref, x_star=x_ref
    )

    Q = model.construct_Q_conditional(theta_ref)
    Qinv = xp.linalg.inv(Q.todense())
    # print_msg("Qinv : \n", Qinv)

    print_msg("marginal variances latent parameters: ", var_latent_params)
    print_msg("ref: marginal variances latent param: ", np.diag(Qinv))

    var_obs = pyinla.get_marginal_variances_observations(theta=theta_ref, x_star=x_ref)
    print_msg("marginal variances observations: ", var_obs[:10])
    var_obs_ref = extract_diagonal(model.a @ Qinv @ model.a.T)
    print_msg("shape var_obs_ref: ", var_obs_ref.shape)
    print_msg("ref: marginal variances observations: ", var_obs_ref[:10])
    print_msg("norm(var_obs - var_obs_ref): ", np.linalg.norm(var_obs - var_obs_ref))

    # call everything
    results = pyinla.run()

    print_msg("results['theta']: ", results["theta"])
    # print_msg("results['f']: ", results["f"])
    # print_msg("results['grad_f']: ", results["grad_f"])
    print_msg("cov_theta: ", results["cov_theta"])
    print_msg(
        "mean of the fixed effects: ",
        results["x"][-model.submodels[-1].n_fixed_effects :],
    )
    print_msg(
        "marginal variances of the fixed effects: ",
        results["marginal_variances_latent"][-model.submodels[-1].n_fixed_effects :],
    )

    print_msg(
        "norm(theta - theta_ref): ",
        np.linalg.norm(results["theta"] - get_host(theta_ref)),
    )
    print_msg("norm(x - x_ref): ", np.linalg.norm(results["x"] - get_host(x_ref)))
    print_msg("Finished.")
