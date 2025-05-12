import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np

from pyinla import xp, backend_flags
from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from pyinla.utils import get_host
from examples_utils.parser_utils import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    print("--- Example: Gaussian spatio-temporal model with regression ---")
    args = parse_args()

    # Configurations of the submodels
    # . Spatio-temporal submodel
    spatio_temporal_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": -2.29160082e-07,
        "r_t": -8.19440054e-07,
        "sigma_st": -0.01996371,
        "manifold": "sphere",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
    }
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_dict),
    )
    # . Regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_regression",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    # Configurations of the likelihood
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 4,
        # "prior_hyperparameters": {"type": "gaussian", "mean": 1.4, "precision": 0.5},
        "prior_hyperparameters": {
            "type": "penalized_complexity",
            "alpha": 0.01,
            "u": 4,
        },
    }

    # Creation of the model by combining the submodels and the likelihood
    model = Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print(model)

    # Configurations of PyINLA
    pyinla_dict = {
        "solver": {"type": "serinv"},
        # "solver": {"type": "dense"},
        "minimize": {
            "max_iter": 200,
            "gtol": 1e-3,
            "disp": True,
            "maxcor": len(model.theta),
        },
        "f_reduction_tol": 1e-3,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    pyinla = PyINLA(
        model=model,
        config=pyinla_config.parse_config(pyinla_dict),
    )

    # Run the optimization
    minimization_result = pyinla.minimize()

    exit()




    # print("\n--- PyINLA results ---")
    # print("Final theta: ", minimization_result["theta"])
    # print("Final f:", minimization_result["f"])
    # print("final grad_f:", minimization_result["grad_f"])

    # print("\n--- References ---")
    theta_ref = xp.array(np.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy"))
    x_ref = xp.array(np.load(f"{BASE_DIR}/reference_outputs/x_ref.npy"))

    # print("theta_ref: ", theta_ref)
    # print(
    #     "Norm between thetas and theta_ref: ",
    #     np.linalg.norm(minimization_result["theta"] - theta_ref),
    # )
    # print(
    #     "Norm between x and x_ref: ", np.linalg.norm(minimization_result["x"] - x_ref)
    # )
    # print("theta_ref: ", theta_ref)

    # Q = model.construct_Q_conditional(theta_ref)
    # Qinv = xp.linalg.inv(Q.todense())
    # # print("Qinv : \n", Qinv)

    # print("marginal variances latent parameters: ", var_latent_params[:10])
    # print("ref: marginal variances latent param: ", np.diag(Qinv)[:10])
    # print(
    #     "norm(var_latent_params - np.diag(Qinv)): ",
    #     np.linalg.norm(var_latent_params - np.diag(Qinv)),
    # )

    # var_obs = pyinla.get_marginal_variances_observations(theta=theta_ref, x_star=x_ref)
    # print("marginal variances observations:      ", var_obs[:10])
    # var_obs_ref = extract_diagonal(model.a @ Qinv @ model.a.T)
    # print("ref: marginal variances observations: ", var_obs_ref[:10])
    # print("norm(var_obs - var_obs_ref): ", np.linalg.norm(var_obs - var_obs_ref))

    # # call everything
    results = pyinla.run()

    if backend_flags["mpi_avail"]:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1

    print("results['theta']: ", results["theta"])
    # print("results['f']: ", results["f"])
    # print("results['grad_f']: ", results["grad_f"])
    print("cov_theta: \n", results["cov_theta"])
    print(
        "mean of the fixed effects: ",
        results["x"][-model.submodels[-1].n_fixed_effects :],
    )
    print(
        "marginal variances of the fixed effects: ",
        results["marginal_variances_latent"][-model.submodels[-1].n_fixed_effects :],
    )

    print(
        f"rank: {rank} norm(theta - theta_ref): {np.linalg.norm(results["theta"] - get_host(theta_ref))}",
    )
    print(
        f"rank: {rank} norm(x - x_ref): ",
        np.linalg.norm(results["x"] - get_host(x_ref)),
    )
