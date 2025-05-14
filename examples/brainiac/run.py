import os

import numpy as np
import scipy.sparse as sp

from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import BrainiacSubModel
from pyinla.utils import scaled_logit

path = os.path.dirname(__file__)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    b = 2  # number of latent variables (number of features)
    m = 2  # number of annotations per feature
    sigma_a2 = 1.0 / 1.0
    precision_mat = sigma_a2 * np.eye(m)

    theta_ref = np.load(f"{base_dir}/inputs_brainiac/theta_original.npy")
    x_ref = np.load(f"{base_dir}/inputs_brainiac/beta_original.npy")

    initial_h2 = theta_ref[0]
    initial_alpha = theta_ref[1:]

    brainiac_dict = {
        "type": "brainiac",
        "input_dir": f"{base_dir}/inputs_brainiac",
        "h2": initial_h2,
        "alpha": initial_alpha,
        "ph_h2": {"type": "beta", "alpha": 5.0, "beta": 1.0},
        "ph_alpha": {
            "type": "gaussian_mvn",
            "mean": np.asarray(theta_ref[1:]),
            "precision": sp.csr_matrix(precision_mat),
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

    # check dimensions
    if model.submodels[0].z.shape[0] != b or model.submodels[0].z.shape[1] != m:
        raise ValueError("Dimension mismatch in Z matrix.")

    print("Model initialized.")

    print("model.theta", model.theta)
    print("length(model.theta)", len(model.theta))
    print("model.theta_keys", model.theta_keys)

    eta = np.ones((model.n_observations, 1))

    model.construct_Q_prior()
    model.construct_Q_conditional(eta)

    # compare to reference solution
    Qprior_ref = sp.load_npz(f"{base_dir}/inputs_brainiac/Qprior_original.npz")
    Qcond_ref = sp.load_npz(f"{base_dir}/inputs_brainiac/Qconditional_original.npz")

    print("Qcond_ref\n", Qcond_ref.toarray())
    print("Qcond\n", model.Q_conditional.toarray())

    print(
        "norm(Qprior_ref - model.Q_prior) = ",
        np.linalg.norm((Qprior_ref - model.Q_prior).toarray()),
    )
    print(
        "norm(Qcond_ref - model.Q_conditional) = ",
        np.linalg.norm((Qcond_ref - model.Q_conditional).toarray()),
    )

    # Q_prior_dense = model.Q_prior.todense()
    # print("Q_prior_dense\n", Q_prior_dense)
    # Q_cond_dense = model.Q_conditional.todense()
    # print("Q_cond_dense\n", Q_cond_dense)

    # plt.matshow(Q_prior_dense)
    # plt.suptitle("Q_prior from brainiac model")
    # plt.savefig("Q_prior.png")

    # plt.matshow(Q_cond_dense)
    # plt.suptitle("Q_conditional from brainiac model")
    # plt.savefig("Q_conditional.png")

    pyinla_dict = {
        # "solver": {"type": "serinv"},
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": 10,
            "gtol": 1e-2,
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

    print("x ref: ", x_ref)
    # minimization_result = pyinla.minimize()

    # output = pyinla._evaluate_f(model.theta)
    # x = model.x
    # print("x: ", x)

    print("\n------ Compare to reference solution ------\n")
    # load reference solution
    # theta_ref = np.load(f"{base_dir}/inputs_brainiac/theta_original.npy")

    # x_ref = np.load(f"{base_dir}/inputs_brainiac/beta_original.npy")
    # x = minimization_result["x"]
    # print("\nx    ", x)
    # print("x_ref", x_ref)
    # print("norm(x_ref - x) = ", np.linalg.norm(x_ref - x))

    results = pyinla.run()

    print("theta_ref: ", theta_ref)
    theta = results["theta"]
    # rescale
    theta[0] = scaled_logit(theta[0], direction="backward")
    print("theta:     ", theta)
    print(
        "norm(theta_ref - minimization_result['theta']) = ",
        np.linalg.norm(theta_ref - results["theta"]),
    )

    # print("results['theta']: ", results["theta"])
    # print("results['f']: ", results["f"])
    # print("results['grad_f']: ", results["grad_f"])
    print("cov_theta: \n", results["cov_theta"])
    print(
        "marginal standard deviations of the hyperparameters: ",
        np.sqrt(results["cov_theta"].diagonal()),
    )
    print("mean of the latent parameters : ", results["x"])
    print(
        "marginal variances of the latent parameters: ",
        results["marginal_variances_latent"],
    )

    print("norm(theta - theta_ref): ", np.linalg.norm(results["theta"] - theta_ref))
    print("norm(x - x_ref): ", np.linalg.norm(results["x"] - x_ref))
