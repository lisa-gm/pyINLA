import os

import numpy as np

from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel
from pyinla.utils import print_msg

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
        "type": "poisson",
        "input_dir": f"{base_dir}",
    }
    model = Model(
        submodels=[regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )

    print_msg(model)

    print_msg("Submodules initialized.")

    pyinla_dict = {
        "solver": {"type": "dense"},
        "minimize": {"max_iter": 50, "gtol": 1e-1, "c1": 1e-4, "c2": 0.9, "disp": True},
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

    print_msg("\n--- PyINLA results ---")
    print_msg("Final theta: ", minimization_result["theta"])
    print_msg("Final x: ", minimization_result["x"])

    print_msg("\n--- References ---")
    x_ref = np.load(f"{base_dir}/reference_outputs/x_ref.npy")
    print_msg("x_ref: ", x_ref)
    print_msg(
        "Norm between x and x_ref: ", np.linalg.norm(minimization_result["x"] - x_ref)
    )
