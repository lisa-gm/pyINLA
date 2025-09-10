import os
import time

# import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as scsp

from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import BrainiacSubModel
from dalia.utils import print_msg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Brainiac Submodel ---")

    base_dir_data = BASE_DIR + "/inputs_brainiac"

    m = 2  # number of annotations per feature
    b = 20000  # number of latent variables / number of features
    sigma_a2 = 1.0 / 1.0
    precision_mat = sigma_a2 * scsp.eye(m)

    theta_ref = xp.load(f"{base_dir_data}/theta_original.npy")
    x_ref = np.load(f"{base_dir_data}/beta_original.npy")

    xp.random.seed(5)
    # has to be between 0 and 1
    initial_h2 = theta_ref[0] - 0.1
    initial_alpha = theta_ref[1:] + 0.5 * xp.random.randn(m - 1)

    brainiac_dict = {
        "type": "brainiac",
        "input_dir": f"{base_dir_data}/inputs_brainiac",
        "h2": initial_h2,
        "alpha": initial_alpha,
        "ph_h2": {"type": "beta", "alpha": 5.0, "beta": 1.0},
        "ph_alpha": {
            "type": "gaussian_mvn",
            "mean": theta_ref[1:],
            "precision": precision_mat,  # sp.sparse.csc_matrix(precision_mat),
        },
    }
    brainiac = BrainiacSubModel(
        config=submodels_config.parse_config(brainiac_dict),
    )
    print(brainiac)

    print("SubModel initialized.")

    likelihood_dict = {"type": "gaussian", "fix_hyperparameters": True}
    model = Model(
        submodels=[brainiac],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )

    print(model)

    print("Model initialized.")

    dalia_dict = {
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": 50,
            "gtol": 1e-3,
            "disp": True,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    tic = time.time()
    result = dalia.run()
    toc = time.time()
    print("Elapsed time dalia.run(): ", toc - tic)

    print("\n------ Compare to reference solution ------\n")
    print("theta_ref: ", theta_ref)

    theta = result["theta_interpret"]
    print("theta_interpret:", theta)

    x = result["x"]
    print("norm(x_ref - x) = ", np.linalg.norm(x_ref - x))

    # marginal variances latent parameters
    var_latent_params = result["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional)
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )
