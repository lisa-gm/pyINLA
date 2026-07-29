import os
import sys

from dalia import xp, sp
from dalia.configs import (
    dalia_config,
    likelihood_config,
    submodels_config,
)
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import AR1SubModel, RegressionSubModel
from dalia.utils import (
    print_msg,
    save_to_json,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: AR1 with Regression and multiple replicates ---")

    save_dalia_results = True  # Set to True to save results to JSON
    
    # Check for parsed parameters
    args = parse_args()

    n_replicates = 2 # number of AR1 replicates
    n_replicates_regression = 2  # number of regression replicates

    # AR1 submodel: replicated
    ar1_dict = {
        "type": "ar1",
        "input_dir": f"{BASE_DIR}/inputs_nrep{n_replicates}/inputs_ar1",
        "n_replicates": n_replicates,
        "phi": 0.5,  # has to be between 0 and 1
        "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
        # initial guess on the precision
        "tau": 3,  # has to be positive
        "ph_tau": {"type": "gamma", "alpha": 2.0, "beta": 1.0},
    }
    ar1 = AR1SubModel(
        config=submodels_config.parse_config(ar1_dict),
    )

    # Regression submodel: single shared intercept (not replicated)
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_nrep{n_replicates}/inputs_regression",
        "n_fixed_effects": 1,
        "n_replicates": n_replicates_regression,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 4.0,
        "prior_hyperparameters": {"type": "gamma", "alpha": 2.0, "beta": 1e-1},
    }
    model = Model(
        submodels=[ar1, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
        input_dir=f"{BASE_DIR}/inputs_nrep{n_replicates}",
    )
    print_msg(model)

    import matplotlib.pyplot as plt

    Qprior = model.construct_Q_prior()
    plt.matshow(Qprior.toarray())
    plt.title("Prior precision matrix Qprior")
    plt.savefig(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/qprior.png")
    plt.close()
    
    A = model.a.toarray()
    plt.matshow(A)
    plt.title("Design matrix A")
    plt.savefig(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/a.png")
    plt.close() 
    
    sp.sparse.save_npz(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/A.npz", model.a)
    sp.sparse.save_npz(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/Qprior.npz", Qprior)
    Qconditional = model.construct_Q_conditional(
        eta=model.a @ model.x
    )
    sp.sparse.save_npz(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/Qconditional.npz", Qconditional)

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )
    
    theta_ref = xp.load(
        f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/theta_ref.npy"
    )
    x_ref = xp.load(f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/x_ref.npy")

    f_value = dalia._evaluate_f(theta_ref)
    
    print(f"f(theta_ref) = {f_value:.4e}")
    #print(f"Gradient at theta_ref: {gradient}")


    results = dalia.run()

    Qconditional = dalia.model.construct_Q_conditional(
        eta=model.a @ results["x"]
    )
    plt.matshow(Qconditional.toarray()[:30, :30])
    plt.title("Conditional precision matrix Qconditional")
    plt.savefig(
        f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/qconditional.png"
    )
    plt.close()

    print_msg("\n--- Results ---")
    print_msg("Theta values external:\n", results["theta"])
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    print_msg("Reference theta:", theta_ref)
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{xp.linalg.norm(results['theta'] - theta_ref):.4e}",
    )

    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref)/ norm(x_ref):   ",
        f"{xp.sqrt(xp.sum((results['x'] - x_ref) ** 2)) / xp.sqrt(xp.sum(x_ref ** 2)):.4e}",
    )

    print_msg("\n--- Marginal distributions of the hyperparameters ---")
    marginals_hp = dalia.marginal_distributions_hp()

    # Extract all hyperparameters
    phi = marginals_hp["hyperparameters"]["phi"]
    tau = marginals_hp["hyperparameters"]["tau"]
    prec_o = marginals_hp["hyperparameters"]["prec_o"]

    print("Quantiles of phi:")
    phi_quantile_pairs = phi["quantiles"]["external"]["pairs"]
    for p, q in phi_quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print("Quantiles of tau:")
    tau_quantile_pairs = tau["quantiles"]["external"]["pairs"]
    for p, q in tau_quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print("Quantiles of prec_o:")
    prec_quantile_pairs = prec_o["quantiles"]["external"]["pairs"]
    for p, q in prec_quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    # save estimates to reference outputs folder
    if save_dalia_results:
        save_to_json(
            results=results,
            filename=f"{BASE_DIR}/inputs_nrep{n_replicates}/reference_outputs/dalia_estimates.json",
        )
