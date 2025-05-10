import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np

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
from pyinla.submodels import RegressionSubModel, SpatialSubModel
from examples_utils.parser_utils import parse_args

SEED = 63
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("--- Example: Gaussian Coregional (3 variates) spatial model with regression ---")

    # Check for parsed parameters
    args = parse_args()
    print("Parsed parameters:")
    print(f"  max_iter: {args.max_iter}")

    nv = 3
    ns = 1092
    nt = 1
    nb = 3
    dim_theta = 12

    n = nv * (ns * nt) + nb

    theta_ref_file = f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_interpretS_original_pyINLA_perm_{dim_theta}_1.dat"
    theta_ref = np.loadtxt(theta_ref_file)

    theta_initial = theta_ref + 0.3 * np.random.randn(dim_theta)

    x_ref_file = f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/x_ref_{n}_1.dat"
    x_ref = xp.loadtxt(x_ref_file)
    print(f"Reference x[-5:]: {x_ref[-5:]}")

    fixed_effects_ref = xp.array([3.0, -8.0, -12.0])

    
    # Configurations of the submodels for the first model
    # . Spatial submodel 1
    spatial_1_dict = {
        "type": "spatial",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_spatial",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0],
        "sigma_e": 0,
        "ph_s": {"type": "gaussian", "mean": theta_ref[0], "precision": 0.5},
        "ph_e": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }
    spatial_1 = SpatialSubModel(
        config=submodels_config.parse_config(spatial_1_dict),
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
        "prec_o": theta_initial[1],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_ref[1],
            "precision": 0.5,
        },
    }
    # Creation of the first model by combining the submodels and the likelihood
    model_1 = Model(
        submodels=[spatial_1, regression_1],
        likelihood_config=likelihood_config.parse_config(likelihood_1_dict),
    )

    # Configurations of the submodels for the second model
    # . Spatial submodel 2
    spatial_2_dict = {
        "type": "spatial",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_spatial",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[2],
        "sigma_e": 0,
        "ph_s": {"type": "gaussian", "mean": theta_ref[2], "precision": 0.5},
        "ph_e": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }
    spatial_2 = SpatialSubModel(
        config=submodels_config.parse_config(spatial_2_dict),
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
        "prec_o": theta_initial[3],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_ref[3],
            "precision": 0.5,
        },
    }
    # Creation of the second model by combining the submodels and the likelihood
    model_2 = Model(
        submodels=[spatial_2, regression_2],
        likelihood_config=likelihood_config.parse_config(likelihood_2_dict),
    )

    # Configurations of the submodels for the third model
    # . Spatial submodel 3
    spatial_3_dict = {
        "type": "spatial",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_3/inputs_spatial",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[4],
        "sigma_e": 0.0,
        "ph_s": {"type": "gaussian", "mean": theta_ref[2], "precision": 0.5},
        "ph_e": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }
    spatial_3 = SpatialSubModel(
        config=submodels_config.parse_config(spatial_3_dict),
    )
    # . Regression submodel 3
    regression_3_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_3/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression_3 = RegressionSubModel(
        config=submodels_config.parse_config(regression_3_dict),
    )
    # . Likelihood submodel 3
    likelihood_3_dict = {
        "type": "gaussian",
        "prec_o": theta_initial[5],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_ref[3],
            "precision": 0.5,
        },
    }
    # Creation of the third model by combining the submodels and the likelihood
    model_3 = Model(
        submodels=[spatial_3, regression_3],
        likelihood_config=likelihood_config.parse_config(likelihood_3_dict),
    )

    # Creation of the coregional model by combining the three models
    coreg_dict = {
        "type": "coregional",
        "n_models": 3,
        "sigmas": [theta_initial[6], theta_initial[7], theta_initial[8]],
        "lambdas": [theta_initial[9], theta_initial[10], theta_initial[11]],
        "ph_sigmas": [
            {"type": "gaussian", "mean": theta_ref[6], "precision": 0.5},
            {"type": "gaussian", "mean": theta_ref[7], "precision": 0.5},
            {"type": "gaussian", "mean": theta_ref[8], "precision": 0.5},
        ],
        "ph_lambdas": [
            {"type": "gaussian", "mean": 0.0, "precision": 0.5},
            {"type": "gaussian", "mean": 0.0, "precision": 0.5},
            {"type": "gaussian", "mean": 0.0, "precision": 0.5},
        ],
    }
    coreg_model = CoregionalModel(
        models=[model_1, model_2, model_3],
        coregional_model_config=models_config.parse_config(coreg_dict),
    )
    print(coreg_model)

    # Configurations of PyINLA
    pyinla_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-3,
            "disp": True,
        },
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



    pyinla.model.theta = xp.array(theta_ref)
    x = xp.zeros_like(pyinla.model.x)
    Qprior = pyinla.model.construct_Q_prior()
    Qcond = pyinla.model.construct_Q_conditional(eta=pyinla.model.a @ x)
    rhs = pyinla.model.construct_information_vector(eta=pyinla.model.a @ x, x_i=x)

    rhs_copy = rhs.copy()
    pyinla.solver.cholesky(Qcond)
    x_est = pyinla.solver.solve(rhs_copy)
    print("x_est: ", x_est[-nb:])
    print("fixed effects ref: ", fixed_effects_ref)

    exit()
    minimization_result = pyinla.minimize()

    print("\n--- PyINLA results ---")

    # print("theta keys: ", coreg_model.theta_keys)
    print("theta ref: ", theta_ref)
    print("minimization_result: ", minimization_result["theta"])

    print("x[-nb:]: ", pyinla.model.x[-nb:])
    print("fixed effects ref: ", fixed_effects_ref)

    # cov_theta = pyinla.compute_covariance_hp(coreg_model.theta)
    # print("covariance hyperparameters(coreg_model.theta): \n", cov_theta)

    # # compute marginals latent parameters
    # theta_ref = xp.array(theta_ref)
    # print("mean latent parameters[-5:]: ", pyinla.model.x[-5:])
    # marginals = pyinla.get_marginal_variances_latent_parameters(
    #     theta=theta_ref, x_star=pyinla.model.x
    # )
    # print("sd latent[-5:]: ", xp.sqrt(marginals[-5:]))

    # marginals_obs = pyinla.get_marginal_variances_observations(
    #     theta=coreg_model.theta, x_star=pyinla.model.x
    # )
    # print("sd obs[:5]: ", xp.sqrt(marginals_obs[:5]))
    # print("mean obs[:5]: ", (coreg_model.a @ pyinla.model.x)[:5])
    # print("y[:5]: ", pyinla.model.y[:5])

    # # call everything
    # results = pyinla.run()

    # print("results['theta']: ", results["theta"])
    # # print("results['f']: ", results["f"])
    # # print("results['grad_f']: ", results["grad_f"])
    # print("cov_theta: \n", results["cov_theta"])
    # print("mean of the fixed effects: ", results["x"][-nb:])
    # print(
    #     "marginal variances of the fixed effects: ",
    #     results["marginal_variances_latent"][-nb:],
    # )

    # print(
    #     "norm(theta - theta_ref): ",
    #     np.linalg.norm(results["theta"] - get_host(theta_ref)),
    # )
    # print("results['x'][-5:]: ", results["x"][-5:])
    # print("norm(x - x_ref): ", np.linalg.norm(results["x"] - get_host(x_ref)))
