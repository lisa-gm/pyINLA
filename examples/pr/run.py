import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel
from pyinla.utils import print_msg
from pyinla import xp
from examples_utils.parser_utils import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Poisson Regression ---")

    # Check for parsed parameters
    args = parse_args()

    # Configurations of the regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )
    # Likelihood
    likelihood_dict = {
        "type": "poisson",
        "input_dir": f"{BASE_DIR}",
    }
    model = Model(
        submodels=[regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )

    print_msg(model)

    # Configurations of PyINLA
    pyinla_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-1,
            "disp": True,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    pyinla = PyINLA(
        model=model,
        config=pyinla_config.parse_config(pyinla_dict),
    )

    minimization_result = pyinla.minimize()

    print_msg("\n--- Results ---")
    print_msg("Theta values:\n", minimization_result["theta"])
    print_msg(
        "Mean of the fixed effects:\n",
        minimization_result["x"][-model.submodels[-1].n_fixed_effects :],
    )

    print_msg("\n--- Comparisons ---")
    # Compare latent parameters
    x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")
    print_msg(
        "Norm (x - x_ref):                ",
        f"{xp.linalg.norm(minimization_result['x'] - x_ref):.4e}",
    )

    print_msg("\n--- Finished ---")
