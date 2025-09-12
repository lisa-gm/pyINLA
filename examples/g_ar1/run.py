import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np

from dalia import xp, sp, backend_flags
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import AR1SubModel, RegressionSubModel
from dalia.utils import get_host, print_msg, scaled_logit  # , extract_diagonal
from examples_utils.parser_utils import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    n = 20

    # load reference output
    theta_original = np.load("reference_outputs/theta_original.npy")
    print(
        "theta original: ",
        theta_original[0],
        xp.log(theta_original[1]),
        xp.log(theta_original[2]),
    )

    x_original = np.load("reference_outputs/x_original.npy")
    print("x original: ", x_original[:10])
    print("dim(x original): ", x_original.shape)

    ar1_dict = {
        "type": "ar1",
        "input_dir": f"{BASE_DIR}/inputs_ar1",
        "n_latent_parameters": n,
        "phi": theta_original[0],  # has to be between 0 and 1
        "tau": xp.log(theta_original[1]),  # assume to already be in log-scale
        "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
        "ph_tau": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
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
        "type": "gaussian",
        "prec_o": xp.log(theta_original[2]),
        "prior_hyperparameters": {
            "type": "penalized_complexity",
            "alpha": 0.01,
            "u": 5,
        },
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

    eta = model.a @ x_original
    Qcond = model.construct_Q_conditional(eta=eta)

    b = model.construct_information_vector(eta=eta, x_i=x_original)

    x_est = np.linalg.solve(Qcond.toarray(), b)
    print("x est: ", x_est)

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
            "maxcor": len(model.theta),
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

    print("theta: ", model.theta)
    # print("x : ", model.x)
    f_value = dalia._evaluate_f(model.theta)
    print("after evaluate f. x: ", model.x)

    results = dalia.minimize()

    theta_unscaled = results["theta"]
    theta = theta_unscaled.copy()
    theta[0] = scaled_logit(theta_unscaled[0], direction="backward")
    theta[1] = np.exp(theta_unscaled[1])
    theta[2] = np.exp(theta_unscaled[2])

    print("theta:          ", theta)
    print("theta original: ", theta_original)

    print("x:          ", results["x"])
    print("x_original: ", x_original)

    print("eta: ", model.a @ x_original)
    print("eta est: ", model.a @ results["x"])
