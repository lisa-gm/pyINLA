import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from examples_utils.parser_utils import parse_args
from pyinla.utils import print_msg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Gaussian spatio-temporal model with regression ---")
    args = parse_args()

    # Configurations of the submodels
    # . Spatio-temporal submodel
    spatio_temporal_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": -0.960279229160082,
        "r_t": -0.3068528194400548,
        "sigma_st": -2.112085713764618,
        "manifold": "sphere",
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
    print_msg(model)

    # Configurations of PyINLA
    pyinla_dict = {
        "solver": {"type": "serinv", "min_processes": args.solver_min_p},
        "minimize": {"max_iter": args.max_iter, "gtol": 1e-1, "disp": True},
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




    print_msg("Final theta: ", minimization_result["theta"])
    print_msg("Final f:", minimization_result["f"])
    print_msg("final grad_f:", minimization_result["grad_f"])
