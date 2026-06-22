from pathlib import Path

import numpy as np

from dalia import xp
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import AR1SubModel, RegressionSubModel
from dalia.utils import print_msg, plot_marginal_distributions_hp, plot_prior_hp  # , extract_diagonal

BASE_DIR: Path = Path(__file__).parent

if __name__ == "__main__":

    np.random.seed(3)

    # load reference output
    theta_original = np.load(BASE_DIR / "reference_outputs" / "theta_original.npy")

    theta_initial = theta_original #[0.6, 1.0, 3.0]
    print("theta initial: ", theta_initial)

    x_original = np.load(BASE_DIR / "reference_outputs" / "x_original.npy")

    ar1_dict = {
        "type": "ar1",
        "input_dir": f"{BASE_DIR}/inputs_ar1",
        "phi": 0.5,  # has to be between 0 and 1
        "ph_phi": {"type": "beta", "alpha": 5.0, "beta": 1.0},
        # initial guess on the precision
        "tau": 3,  # has to be positive
        "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 0.9},
        # initial guess on the variance
        # "sigma2": 0.33, # has to be positive
        # "ph_sigma2": {"type": "invgamma", "alpha": 2.0, "beta": 1.0},
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
        "prec_o": 20,
        # "prior_hyperparameters": {
        #     "type": "penalized_complexity",
        #     "alpha": 0.01,
        #     "u": 5,
        # },
        "prior_hyperparameters": {
            "type": "half_normal",
            "precision": 1e-4,
        },
    }

    model = Model(
        submodels=[ar1, regression], #
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)
    
    import matplotlib.pyplot as plt

    # Plot all three priors
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    # Plot 1: phi (Beta prior)
    phi_interval = np.linspace(1e-6, 1 - 1e-6, 200)
    phi_internal_interval = model.prior_hyperparameters[
        0
    ].rescale_hyperparameters_to_internal(phi_interval, "forward")
    log_prior_phi = model.prior_hyperparameters[0].evaluate_log_prior(phi_interval)
    prior_phi = model.prior_hyperparameters[0].evaluate_prior(phi_interval)
    theta_original_internal = model.prior_hyperparameters[
        0
    ].rescale_hyperparameters_to_internal(theta_original[0], "forward")
    tranformed_prior_phi = model.prior_hyperparameters[0].evaluate_internal_log_prior(
        phi_internal_interval
    )

    # Plot 1a: phi evaluate_prior (Column 0)
    axes[0, 0].plot(phi_interval, prior_phi, "b-", linewidth=2, label="Prior PDF")
    axes[0, 0].axvline(
        theta_original[0],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Reference: {theta_original[0]:.4f}",
    )
    axes[0, 0].set_xlabel("φ (external)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("φ Beta Prior PDF (External Space)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 1b: phi log prior (Column 1)
    axes[0, 1].plot(phi_interval, log_prior_phi, "b-", linewidth=2, label="Log Prior")
    axes[0, 1].axvline(
        theta_original[0],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Reference: {theta_original[0]:.4f}",
    )
    axes[0, 1].set_xlabel("φ (external)")
    axes[0, 1].set_ylabel("Log Density")
    axes[0, 1].set_title("φ Log Prior (External Space)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 1c: phi internal log prior (Column 2)
    axes[0, 2].plot(
        phi_internal_interval,
        tranformed_prior_phi,
        "orange",
        linewidth=2,
        label="Internal Log Prior",
    )
    axes[0, 2].axvline(
        theta_original_internal,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Reference (internal): {theta_original_internal:.4f}",
    )
    axes[0, 2].set_xlabel("φ (internal)")
    axes[0, 2].set_ylabel("Log Density")
    axes[0, 2].set_title("φ Log Prior (Internal Space)")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # Plot 2: tau (Gamma prior)
    tau_interval = np.linspace(0.01, 10, 200)
    tau_internal_interval = model.prior_hyperparameters[
        1
    ].rescale_hyperparameters_to_internal(tau_interval, "forward")
    log_prior_tau = model.prior_hyperparameters[1].evaluate_log_prior(tau_interval)
    prior_tau = model.prior_hyperparameters[1].evaluate_prior(tau_interval)
    tranformed_prior_tau = model.prior_hyperparameters[1].evaluate_internal_log_prior(
        tau_internal_interval
    )
    theta_original_internal_tau = model.prior_hyperparameters[
        1
    ].rescale_hyperparameters_to_internal(theta_original[1], "forward")

    # Plot 2a: tau evaluate_prior (Column 0)
    axes[1, 0].plot(tau_interval, prior_tau, "g-", linewidth=2, label="Prior PDF")
    axes[1, 0].axvline(
        theta_original[1],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Reference: {theta_original[1]:.4f}",
    )
    axes[1, 0].set_xlabel("τ (external)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("τ GAmma Prior PDF (External Space)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 2b: tau log prior (Column 1)
    axes[1, 1].plot(tau_interval, log_prior_tau, "g-", linewidth=2, label="Log Prior")
    axes[1, 1].axvline(
        theta_original[1],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Reference: {theta_original[1]:.4f}",
    )
    axes[1, 1].set_xlabel("τ (external)")
    axes[1, 1].set_ylabel("Log Density")
    axes[1, 1].set_title("τ Log Prior (External Space)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Plot 2c: tau internal log prior (Column 2)
    axes[1, 2].plot(
        tau_internal_interval,
        tranformed_prior_tau,
        "orange",
        linewidth=2,
        label="Internal Log Prior",
    )
    axes[1, 2].axvline(
        theta_original_internal_tau,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Reference (internal): {theta_original_internal_tau:.4f}",
    )
    axes[1, 2].set_xlabel("τ (internal)")
    axes[1, 2].set_ylabel("Log Density")
    axes[1, 2].set_title("τ Log Prior (Internal Space)")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    # Plot 3: prec_o (Half-normal prior)
    # Plot 3: prec_o (Half-normal prior)
    prec_o_interval = np.linspace(0.01, 100, 200)
    prec_o_internal_interval = model.prior_hyperparameters[
        2
    ].rescale_hyperparameters_to_internal(prec_o_interval, "forward")
    log_prior_prec_o = model.prior_hyperparameters[2].evaluate_log_prior(
        prec_o_interval
    )
    prior_prec_o = model.prior_hyperparameters[2].evaluate_prior(prec_o_interval)
    transformed_prior_prec_o = model.prior_hyperparameters[
        2
    ].evaluate_internal_log_prior(prec_o_internal_interval)
    theta_original_internal_prec_o = model.prior_hyperparameters[
        2
    ].rescale_hyperparameters_to_internal(theta_original[2], "forward")

    # Plot 3a: prec_o evaluate_prior (Column 0)
    axes[2, 0].plot(prec_o_interval, prior_prec_o, "m-", linewidth=2, label="Prior PDF")
    axes[2, 0].axvline(
        theta_original[2],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Initial: {theta_original[2]:.4f}",
    )
    axes[2, 0].set_xlabel("variance (external)")
    axes[2, 0].set_ylabel("Density")
    axes[2, 0].set_title("prec_o Half Normal Prior PDF (External Space)")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)

    # Plot 3b: prec_o log prior (Column 1)
    axes[2, 1].plot(
        prec_o_interval, log_prior_prec_o, "m-", linewidth=2, label="Log Prior"
    )
    axes[2, 1].axvline(
        theta_original[2],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Initial: {theta_original[2]:.4f}",
    )
    axes[2, 1].set_xlabel("variance (external)")
    axes[2, 1].set_ylabel("Log Density")
    axes[2, 1].set_title("prec_o Log Prior (External Space)")
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)

    # Plot 3c: prec_o internal log prior (Column 2)
    axes[2, 2].plot(
        prec_o_internal_interval,
        transformed_prior_prec_o,
        "cyan",
        linewidth=2,
        label="Internal Log Prior",
    )
    axes[2, 2].axvline(
        theta_original_internal_prec_o,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Reference (internal): {theta_original_internal_prec_o:.4f}",
    )
    axes[2, 2].set_xlabel("variance (internal)")
    axes[2, 2].set_ylabel("Log Density")
    axes[2, 2].set_title("prec_o Log Prior (Internal Space)")
    axes[2, 2].legend()
    axes[2, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    exit()

    Qprior = model.construct_Q_prior()
    Qinv = xp.linalg.inv(Qprior.toarray())

    # in gaussian case x = 0, thus eta = 0
    x_i = xp.zeros(model.n_latent_parameters)
    eta = model.a @ x_i
    Qcond = model.construct_Q_conditional(eta=eta)

    b = model.construct_information_vector(eta=eta, x_i=x_i)

    x_est = xp.linalg.solve(Qcond.toarray(), b)
    # print("x est: ", x_est)
    print("norm(x_original - x_est): ", xp.linalg.norm(xp.asarray(x_original) - x_est))

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
        "verbosity": 0,
    }

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    print("initial model theta: ", model.theta_external)

    print("\nCalling DALIA.run()")
    results = dalia.run()

    print_msg("\n--- Results ---")

    theta = results["theta"]
    print("theta:          ", np.round(theta, 4))
    print("theta original: ", theta_original)

    print_msg("Covariance of theta:\n", results["cov_theta_internal"])
    print_msg(
        "Mean of the fixed effects:\n",
        results["x"][-model.submodels[-1].n_fixed_effects :],
    )

    # print("x:          ", results["x"])
    # print("x_original: ", x_original)

    # print("eta: ", model.a @ x_original)
    # print("eta est: ", model.a @ results["x"])

    print_msg("\n--- Comparisons ---")
    print("norm(eta - eta_est): ", xp.linalg.norm(model.a @ xp.asarray(x_original) - model.a @ results["x"]))
    print("normalized norm(eta - eta_est): ", xp.linalg.norm(model.a @ xp.asarray(x_original) - model.a @ results["x"]) / xp.linalg.norm(model.a @ xp.asarray(x_original)))

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{xp.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    print_msg("\n--- Marginal distributions of the hyperparameters ---")
    marginals_hp = dalia.marginal_distributions_hp() 

    fig, axes = plot_marginal_distributions_hp(marginals_hp)
    import matplotlib.pyplot as plt
    plt.show()

    phi = marginals_hp["hyperparameters"]["prec_o"]
    quantile_pairs = phi['quantiles']['external']['pairs']

    print("Quantile pairs of phi:")
    for p, q in quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    print_msg("\n--- Finished ---")
