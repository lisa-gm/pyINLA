import os
import argparse

from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from examples.examples_utils.parser_utils import parse_args


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("--- Example: Gaussian spatio-temporal model with regression ---")

    # Check for parsed parameters
    parser = argparse.ArgumentParser(description="PyINLA example parameters")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of iterations in the optimization process.",
    )
    parser.add_argument(
        "--solver_min_p",
        type=int,
        default=1,
        help="Minimum number of processes for the solver. If greater than 1 a distributed solver is used.",
    )
    args = parser.parse_args()
    print("Parsed parameters:")
    print(f"  max_iter: {args.max_iter}")
    print(f"  solver_min_p: {args.solver_min_p}")

    # Configurations of the submodels
    # . Spatio-temporal submodel
    spatio_temporal_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0.03972077083991806,
        "r_t": 1.6931471805599454,
        "sigma_st": 1.8879142862353822,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": 0.03972077083991806, "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": 2.3931471805599456, "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 1.4379142862353824, "precision": 0.5},
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
        "prec_o": 0.5,
        "prior_hyperparameters": {"type": "gaussian", "mean": 1.4, "precision": 0.5},
    }

    # Creation of the model by combining the submodels and the likelihood
    model = Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print(model)

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




    # print("Final theta: ", minimization_result["theta"])
    # print("Final f:", minimization_result["f"])
    # print("final grad_f:", minimization_result["grad_f"])
    # results = pyinla.run()
    # print("results['theta']: ", results["theta"])
    # print("cov_theta: \n", results["cov_theta"])
    # print(
    #     "mean of the fixed effects: ",
    #     results["x"][-model.submodels[-1].n_fixed_effects :],
    # )
    # print(
    #     "marginal variances of the fixed effects: ",
    #     results["marginal_variances_latent"][-model.submodels[-1].n_fixed_effects :],
    # )
