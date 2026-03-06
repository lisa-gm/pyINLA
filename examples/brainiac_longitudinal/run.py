import os
import time

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from dalia import xp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import BrainiacSubModel, RegressionSubModel
from dalia.utils import scaled_logit

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Parameters — must match generate_data.py
    n_subjects    = 2000
    t_per_subject = 5
    n_obs         = n_subjects * t_per_subject
    b             = 12000   # genetic effect dimension
    m             = 2       # annotation dimension

    # =============================================
    # Load reference outputs
    # =============================================
    theta_ref = xp.load(os.path.join(base_dir, "reference_outputs", "theta_original.npy"))
    x_ref     = np.load(os.path.join(base_dir, "reference_outputs", "beta_original.npy"))

    # =============================================
    # Build submodels
    # =============================================
    brainiac_dict = {
        "type": "brainiac",
        "input_dir": os.path.join(base_dir, "inputs_brainiac"),
        "h2": theta_ref[0],
        "alpha": theta_ref[1:],
        "ph_h2": {"type": "beta", "alpha": 1.0, "beta": 1.0},
        "ph_alpha": {
            "type": "gaussian_mvn",
            "mean": xp.zeros(m),
            "precision": 0.01 * sp.eye(m),
        },
    }
    brainiac = BrainiacSubModel(config=submodels_config.parse_config(brainiac_dict))

    trend_submodel = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": os.path.join(base_dir, "inputs_trend"),
        "n_fixed_effects": 2,
        "fixed_effects_prior_precision": 1e-3,
    }))

    subject_submodel = RegressionSubModel(config=submodels_config.parse_config({
        "type": "regression",
        "input_dir": os.path.join(base_dir, "inputs_subjects"),
        "n_fixed_effects": n_subjects,
        "fixed_effects_prior_precision": 1.0,
    }))

    # =============================================
    # Assemble model and DALIA solver
    # =============================================
    model = Model(
        submodels=[brainiac, trend_submodel, subject_submodel],
        likelihood_config=likelihood_config.parse_config({
            "type": "gaussian",
            "fix_hyperparameters": True,
        }),
    )

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config({
            "solver": {"type": "dense"},
            "minimize": {"max_iter": 50, "gtol": 1e-3, "disp": True},
            "simulation_dir": os.path.join(base_dir, "dalia_cache"),
        }),
    )

    # =============================================
    # Run
    # =============================================
    t_start = time.time()
    result  = dalia.run()
    t_elapsed = time.time() - t_start
    print(f"\nDALIA running time: {t_elapsed:.2f} seconds")

    # =============================================
    # Hyperparameter comparison
    # =============================================
    theta_raw = result["theta"]
    h2_est    = scaled_logit(theta_raw[0], direction="backward")
    alpha_est = theta_raw[1:]

    print("\n------ Hyperparameter Comparison ------")
    print(f"  h2   : estimated = {h2_est:.4f}  |  reference = {theta_ref[0]:.4f}")
    for i, (a_est, a_ref) in enumerate(zip(alpha_est, theta_ref[1:])):
        print(f"  alpha_{i}: estimated = {a_est:.4f}  |  reference = {a_ref:.4f}")

    # =============================================
    # Beta recovery analysis
    # =============================================
    x_beta    = result["x"][:b]    # first b entries correspond to brainiac's beta
    x_ref_arr = x_ref

    norm_diff = np.linalg.norm(x_ref_arr - x_beta)
    norm_ref  = np.linalg.norm(x_ref_arr)
    rel_err   = norm_diff / (norm_ref + 1e-12)
    corr      = np.corrcoef(x_ref_arr, x_beta)[0, 1]

    print("\n------ Beta Recovery ------")
    print(f"  ||beta_ref||                      = {norm_ref:.4e}")
    print(f"  ||beta_est - beta_ref||           = {norm_diff:.4e}")
    print(f"  Relative error                    = {rel_err:.4%}")
    print(f"  Pearson corr(beta_est, beta_ref)  = {corr:.4f}")

    # =============================================
    # Visualization
    # =============================================
    residuals = x_beta - x_ref_arr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Estimated vs Reference
    ax = axes[0]
    ax.scatter(x_ref_arr, x_beta, color="steelblue", alpha=0.75, s=40,
               label=f"Pearson r = {corr:.3f}")
    lims = [min(x_beta.min(), x_ref_arr.min()) - 0.05,
            max(x_beta.max(), x_ref_arr.max()) + 0.05]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y = x)")
    ax.set_xlabel("Reference Beta")
    ax.set_ylabel("Estimated Beta")
    ax.set_title("Beta Recovery: Estimated vs Reference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Residual bar chart
    ax = axes[1]
    ax.bar(range(len(residuals)), residuals, color="salmon", alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Beta Index")
    ax.set_ylabel("Residual (Estimated - Reference)")
    ax.set_title(f"Beta Residuals  |  ||delta|| = {norm_diff:.4e}  |  Rel Err = {rel_err:.2%}")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"N={n_subjects}, T={t_per_subject}, b={b}, m={m} | "
        f"h2: est={h2_est:.4f} ref={theta_ref[0]:.4f} | "
        f"time={t_elapsed:.1f}s",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "beta_recovery.png"))
    plt.show()
