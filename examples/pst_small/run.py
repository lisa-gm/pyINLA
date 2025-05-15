import os

import numpy as np

from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from pyinla.utils import print_msg

path = os.path.dirname(__file__)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    spatio_temporal_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{base_dir}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0,
        "r_t": 0,
        "sigma_st": 2,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": -2.30258509299405, "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": 0.693147180559945, "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 1.38629436111989, "precision": 0.5},
    }
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_dict),
    )

    regression_dict = {
        "type": "regression",
        "input_dir": f"{base_dir}/inputs_regression",
        "n_fixed_effects": 8,
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
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )

    print_msg(model)

    print_msg("Submodules initialized.")

    pyinla_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": 100,
            "gtol": 1e-3,
            "c1": 1e-4,
            "c2": 0.9,
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

    print_msg("\n--- PyINLA results ---")
    print_msg("Final theta: ", minimization_result["theta"])
    print_msg("Final f:", minimization_result["f"])
    print_msg("final grad_f:", minimization_result["grad_f"])

    print_msg("\n--- References ---")
    theta_ref = np.load(f"{base_dir}/reference_outputs/theta_ref.npy")
    x_ref = np.load(f"{base_dir}/reference_outputs/x_ref.npy")

    print_msg("theta_ref: ", theta_ref)
    print_msg(
        "Norm between thetas and theta_ref: ",
        np.linalg.norm(minimization_result["theta"] - theta_ref),
    )
    print_msg(
        "Norm between x and x_ref: ", np.linalg.norm(minimization_result["x"] - x_ref)
    )
