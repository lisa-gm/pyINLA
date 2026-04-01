import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np

from dalia import xp
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import AR1SubModel, RegressionSubModel
from dalia.utils import get_host, print_msg  # , extract_diagonal


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    n = 1000

    # load reference output
    theta_original = np.load("reference_outputs/theta_original.npy")
    print("theta original: ", theta_original)

    x_original = np.load("reference_outputs/x_original.npy")
    print("x original: ", x_original[:10])
    print("dim(x original): ", x_original.shape)

    ar1_dict = {
        "type": "ar1",
        "input_dir": f"{BASE_DIR}/inputs_ar1",
        "phi": 0.45,  # has to be between 0 and 1
        "tau": 0.5,  # precision 
        "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 0.5},
    }
    ar1 = AR1SubModel(
        config=submodels_config.parse_config(ar1_dict),
    )
    
    # Configurations of the regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    likelihood_dict = {
        "type": "poisson",
        "input_dir": f"{BASE_DIR}",
    }

    model = Model(
        submodels=[ar1, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    Qprior = model.construct_Q_prior()
    print("Qprior: \n", Qprior.toarray())
    Qinv = np.linalg.inv(Qprior.toarray())
    geom_mean = np.exp(np.mean(np.log(Qinv.diagonal())))
    print("Geometric mean of Qinv diagonal: ", geom_mean)

    eta = model.a @ model.x
    Qcond = model.construct_Q_conditional(eta=eta)

    # L = np.linalg.cholesky(Qcond.toarray())

    # plt.spy(Qcond, markersize=2)
    # plt.title("Sparsity pattern of Qcond")
    # plt.show()

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": 100,
            "gtol": 1e-3,
            "disp": True,
            "maxcor": len(model.theta_external),
        },
        "f_reduction_tol": 1e-3,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    print("theta external: ", model.theta_external)
    # print("x : ", model.x)
    f_value = dalia._evaluate_f(model.theta_external)
    print("after evaluate f. x: ", model.x)

    results = dalia.minimize()

    theta_unscaled = results["theta"]
    print("theta unscaled: ", theta_unscaled)
    theta_original_log = xp.array(
        [
            xp.asarray(theta_original[0]),
            xp.log(theta_original[1]),
        ]
    )
    print("theta original log: ", theta_original_log)

    print("norm(x_original - x): ", np.linalg.norm(x_original - get_host(results["x"])))
    print("normalized norm: ", np.linalg.norm(x_original - get_host(results["x"])) / np.linalg.norm(x_original))

    # print("norm(eta_original - eta_est): ", np.linalg.norm(model.a @ x_original - model.a @ get_host(results["x"])))
    # print(
    #     "normalized norm: ",
    #     np.linalg.norm(model.a @ x_original - model.a @ get_host(results["x"])) / np.linalg.norm(model.a @ x_original),
    # )
