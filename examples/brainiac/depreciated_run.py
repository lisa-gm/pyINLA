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
from dalia.utils import plot_marginal_distributions_hp, print_msg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Brainiac Submodel ---")

    # base_dir_data = BASE_DIR + "/inputs_brainiac_cmPRS"
    base_dir_data = BASE_DIR

    m = 2  # number of annotations per feature
    b = 20  # number of latent variables / number of features
    sigma_a2 = 1.0 / 1.0
    precision_mat = sigma_a2 * scsp.eye(m)

    theta_ref = xp.load(f"{base_dir_data}/reference_outputs/theta_original.npy")
    x_ref = np.load(f"{base_dir_data}/reference_outputs/beta_original.npy")

    print("Reference theta: ", theta_ref)

    xp.random.seed(5)
    # has to be between 0 and 1
    initial_h2 = theta_ref[0] - 0.1
    initial_alpha = theta_ref[1:] + 1.5 * xp.random.randn(m)

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

    theta = result["theta"]
    print("theta dalia:", theta)

    x = result["x"]
    print("norm(x_ref - x) = ", np.linalg.norm(x_ref - x))

    # marginal variances latent parameters
    var_latent_params = result["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)

    if scsp.issparse(Qconditional):
        Q_inv_ref = xp.linalg.inv(Qconditional.toarray())
    else:
        Q_inv_ref = xp.linalg.inv(Qconditional)
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Q_inv_ref)):.4e}",
    )

    marginals_hyperparameters = dalia.marginal_distributions_hp()

    fig, axes = plot_marginal_distributions_hp(marginals_hyperparameters)
    import matplotlib.pyplot as plt

    plt.savefig("marginal_distributions_hp_brainiac.png")

    h2 = marginals_hyperparameters["hyperparameters"]["h2"]
    quantile_pairs = h2["quantiles"]["external"]["pairs"]

    print("Quantile pairs of phi:")
    for p, q in quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print_msg("\n--- Finished ---")
