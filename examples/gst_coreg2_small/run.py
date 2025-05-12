import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import time

from pyinla import xp
from pyinla.configs import (
    likelihood_config,
    models_config,
    pyinla_config,
    submodels_config,
)
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.models import CoregionalModel
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from pyinla.utils import get_host, print_msg
from examples_utils.parser_utils import parse_args
from examples_utils.infos_utils import summarize_sparse_matrix

SEED = 63
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Gaussian Coregional (2 variates) spatio-temporal model with regression ---")

    # Check for parsed parameters
    args = parse_args()

    nv = 2
    ns = 354
    nt = 12
    nb = 2
    dim_theta = 9

    n = nv * ns * nt + nb

    theta_ref_file = (
        f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_interpretS_original_pyINLA_perm_{dim_theta}_1.dat"
        # f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_interpretS_original_pyINLA_perm_{dim_theta}_1.npy"
    )
    theta_ref = np.loadtxt(theta_ref_file)

    perturbation = [
        0.18197867,
        -0.12551227,
        0.19998896,
        0.17226796,
        0.14656176,
        -0.11864931,
        0.17817371,
        -0.13006157,
        0.19308036,
    ]

    theta_initial = theta_ref + np.array(perturbation)

    print_msg(f"theta_initial: {theta_initial}")

    fixed_effects_ref = xp.array([-3.0, 8.0])

    # Configurations of the submodels for the first model
    # . Spatio-temporal submodel 1
    spatio_temporal_1_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0],
        "r_t": theta_initial[1],
        "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {
            "type": "gaussian", 
            "mean": theta_ref[0], 
            "precision": 0.5,
        },
        "ph_t": {
            "type": "gaussian", 
            "mean": theta_ref[1], 
            "precision": 0.5,
        },
        "ph_st": {
            "type": "gaussian", 
            "mean": 0.0, 
            "precision": 0.5,
        },
    }
    spatio_temporal_1 = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_1_dict),
    )
    # . Regression submodel 1
    regression_1_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression_1 = RegressionSubModel(
        config=submodels_config.parse_config(regression_1_dict),
    )
    # . Likelihood submodel 1
    likelihood_1_dict = {
        "type": "gaussian",
        "prec_o": theta_initial[2],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_initial[2],
            "precision": 0.5,
        },
    }
    # Creation of the first model by combining the submodels and the likelihood
    model_1 = Model(
        submodels=[regression_1, spatio_temporal_1],
        likelihood_config=likelihood_config.parse_config(likelihood_1_dict),
    )

    # Configurations of the submodels for the second model
    # . Spatio-temporal submodel 2
    spatio_temporal_2_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[3],
        "r_t": theta_initial[4],
        "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {
            "type": "gaussian", 
            "mean": theta_ref[3], 
            "precision": 0.5,
        },
        "ph_t": {
            "type": "gaussian", 
            "mean": theta_ref[4], 
            "precision": 0.5,
        },
        "ph_st": {
            "type": "gaussian", 
            "mean": 0.0, 
            "precision": 0.5,
        },
    }
    spatio_temporal_2 = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_2_dict),
    )
    # . Regression submodel 2
    regression_2_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression_2 = RegressionSubModel(
        config=submodels_config.parse_config(regression_2_dict),
    )
    # . Likelihood submodel 2
    likelihood_2_dict = {
        "type": "gaussian",
        "prec_o": theta_initial[5],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_ref[5],
            "precision": 0.5,
        },
    }
    # Creation of the second model by combining the submodels and the likelihood
    model_2 = Model(
        submodels=[spatio_temporal_2, regression_2],
        likelihood_config=likelihood_config.parse_config(likelihood_2_dict),
    )

    # Creation of the coregional model by combining the models
    coreg_dict = {
        "type": "coregional",
        "n_models": 2,
        "sigmas": [theta_initial[6], theta_initial[7]],
        "lambdas": [theta_initial[8]],
        "ph_sigmas": [
            {"type": "gaussian", "mean": theta_ref[6], "precision": 0.5},
            {"type": "gaussian", "mean": theta_ref[7], "precision": 0.5},
        ],
        "ph_lambdas": [
            {"type": "gaussian", "mean": 0.0, "precision": 0.5},
        ],
    }
    coreg_model = CoregionalModel(
        models=[model_1, model_2],
        coregional_model_config=models_config.parse_config(coreg_dict),
    )
    print_msg(coreg_model)

    # Configurations of PyINLA
    pyinla_dict = {
        "solver": {
            "type": "serinv",
            "min_processes": args.solver_min_p,
        },
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-3,
            "disp": True,
            "maxcor": len(coreg_model.theta),
        },
        "f_reduction_tol": 1e-3,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "eps_hessian_f": 5 * 1e-3,
        "simulation_dir": ".",
    }
    pyinla = PyINLA(
        model=coreg_model,
        config=pyinla_config.parse_config(pyinla_dict),
    )

    # Run the optimization
    minimization_result = pyinla.minimize()

    exit()
    # TODO: From here, to be curated



    print_msg("\nCalling PyINLA.minimize()")
    tic = time.perf_counter()
    results = pyinla.minimize()
    toc = time.perf_counter()
    # print_msg("f_value: ", f_value)
    print_msg(f"PyINLA.minimize() took {toc - tic:0.4f} seconds")

    print_msg("\n--- PyINLA results ---")
    print_msg("theta initial: ", theta_initial)
    print_msg("theta ref:     ", theta_ref)

    print_msg(
        f"norm(theta - theta_ref): {np.linalg.norm(results["theta"] - get_host(theta_ref))}",
    )

    print_msg("FE ref:  ", fixed_effects_ref)
    print_msg(f"x[-nb:]: , {pyinla.model.x[-nb:]}")

    # pyinla.model.theta = xp.array(theta_ref)
    # x = xp.zeros_like(pyinla.model.x)

    # pyinla.solver.cholesky(Qprior, sparsity="bt")
    # logdet_Qprior = pyinla.solver.logdet(sparsity="bt")
    # print_msg("logdet_Qprior: ", logdet_Qprior)

    # Qcond = pyinla.model.construct_Q_conditional(eta=pyinla.model.a @ x)
    # summarize_sparse_matrix(Qcond, "Qcond")

    # pyinla.solver.cholesky(Qcond, sparsity="bta")
    # logdet_Qcond = pyinla.solver.logdet(sparsity="bta")
    # print_msg("logdet_Qcond: ", logdet_Qcond)

    # rhs = pyinla.model.construct_information_vector(eta=pyinla.model.a @ x, x_i=x)
    # print_msg(f"Min: {rhs.min():.6f}, Max: {rhs.max():.6f}")
    # print_msg(f"Mean: {rhs.mean():.6f}, Std: {rhs.std():.6f}")

    # rhs_copy = rhs.copy()
    # x_est = pyinla.solver.solve(rhs_copy, "bta")
    # print_msg("x_est:  ", x_est[-nb:])
    # print_msg("FE ref: ", fixed_effects_ref)

    # minimization_result = pyinla.minimize()

    # print_msg("\n--- PyINLA results ---")

    # # print_msg("theta keys: ", coreg_model.theta_keys)
    # print_msg("theta initial: ", theta_initial)
    # print_msg("theta ref:           ", theta_ref)
    # print_msg("minimization_result: ", minimization_result["theta"])

    # print_msg("x[-nb:]: ", pyinla.model.x[-nb:])
    # print_msg("FE ref:  ", fixed_effects_ref)

    # exit()
    # cov_theta = pyinla.compute_covariance_hp(coreg_model.theta)
    # print_msg("covariance hyperparameters(coreg_model.theta): \n", cov_theta)

    # # compute marginals latent parameters
    # print_msg("mean latent parameters[-5:]: ", pyinla.model.x[-5:])
    # marginals = pyinla.get_marginal_variances_latent_parameters()
    # print_msg("sd latent[-5:]: ", np.sqrt(marginals[-5:]))

    # marginals_obs = pyinla.get_marginal_variances_observations(
    #     theta=coreg_model.theta, x_star=pyinla.model.x
    # )
    # print_msg("sd obs[:5]: ", np.sqrt(marginals_obs[:5]))
    # print_msg("mean obs[:5]: ", (coreg_model.a @ pyinla.model.x)[:5])
    # print_msg("y[:5]: ", pyinla.model.y[:5])

    # # call everything
    # results = pyinla.run()

    # print_msg("results['theta']: ", results["theta"])
    # # print_msg("results['f']: ", results["f"])
    # # print_msg("results['grad_f']: ", results["grad_f"])
    # print_msg("cov_theta: \n", results["cov_theta"])
    # print_msg("mean of the fixed effects: ", results["x"][-nb:])
    # print_msg(
    #     "marginal variances of the fixed effects: ",
    #     results["marginal_variances_latent"][-nb:],
    # )

    # print_msg(
    #     "norm(theta - theta_ref): ",
    #     np.linalg.norm(results["theta"] - get_host(theta_ref)),
    # )
    # # print_msg("results['x'][-5:]: ", results["x"][-5:])
    # # print_msg("norm(x - x_ref): ", np.linalg.norm(results["x"] - get_host(x_ref)))
