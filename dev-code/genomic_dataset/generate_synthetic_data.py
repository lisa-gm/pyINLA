import os
from pathlib import Path

import numpy as np
from scipy import sparse


BASE_DIR = Path(__file__).resolve().parent


def make_balanced_group_indices(n_obs: int, n_groups: int) -> np.ndarray:
    """Create near-balanced group assignments and shuffle them."""
    base = np.repeat(np.arange(n_groups), n_obs // n_groups)
    remainder = np.arange(n_obs % n_groups)
    groups = np.concatenate([base, remainder])
    rng = np.random.default_rng(123)
    rng.shuffle(groups)
    return groups


def main() -> None:
    rng = np.random.default_rng(41)

    # Keep model structure identical to run.py, but generate stable synthetic data.
    n_obs = 4000

    n_iid = 40
    n_dense = 40
    n_fixed = 10

    tau_iid_true = 3.0
    tau_dense_true = 10.0
    prec_o_true = 50.0

    # Directories expected by run.py
    inputs_iid_dir = BASE_DIR
    inputs_dense_dir = BASE_DIR
    inputs_fixed_dir = BASE_DIR
    ref_dir = BASE_DIR / "ref_dir"

    for p in [inputs_iid_dir, inputs_dense_dir, inputs_fixed_dir, ref_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # 1) IID generic component
    group_idx = make_balanced_group_indices(n_obs, n_iid)
    rows = np.arange(n_obs)
    cols = group_idx
    data = np.ones(n_obs, dtype=float)
    iid_design = sparse.coo_matrix((data, (rows, cols)), shape=(n_obs, n_iid)).tocsc()
    iid_prior = sparse.identity(n_iid, format="csc")

    # True latent effects
    u_iid = rng.normal(loc=0.0, scale=np.sqrt(1.0 / tau_iid_true), size=n_iid)

    # 2) Dense generic component: random design and SPD precision matrix.
    queen_design = rng.normal(loc=0.0, scale=1.0 / np.sqrt(n_dense), size=(n_obs, n_dense))

    m = rng.normal(loc=0.0, scale=0.2, size=(n_dense, n_dense))
    queen_prior_np = m.T @ m + np.eye(n_dense)
    queen_prior = sparse.csc_matrix(queen_prior_np)

    cov_dense = np.linalg.inv(tau_dense_true * queen_prior_np)
    l_dense = np.linalg.cholesky(cov_dense)
    u_dense = l_dense @ rng.normal(size=n_dense)

    # 3) Regression component: intercept + near-orthonormal covariates.
    x_raw = rng.normal(size=(n_obs, n_fixed - 1))
    q_cov, _ = np.linalg.qr(x_raw, mode="reduced")
    regression_design = np.column_stack([np.ones(n_obs), q_cov])

    # Sparse signal setting: only a few fixed effects are nonzero.
    beta_true = np.zeros(n_fixed)
    beta_true[0] = 0.5
    n_nonzero = 8
    nonzero_idx = rng.choice(np.arange(1, n_fixed), size=n_nonzero, replace=False)
    beta_true[nonzero_idx] = rng.normal(loc=0.0, scale=0.2, size=n_nonzero)

    eta = iid_design @ u_iid + queen_design @ u_dense + regression_design @ beta_true
    observations = eta + rng.normal(loc=0.0, scale=np.sqrt(1.0 / prec_o_true), size=n_obs)

    # Save run.py inputs
    np.save(BASE_DIR / "observations.npy", observations)

    np.save(inputs_iid_dir / "iid_design.npy", iid_design.toarray())
    np.save(inputs_iid_dir / "iid_prior.npy", iid_prior.toarray())

    np.save(inputs_dense_dir / "queen_design.npy", queen_design)
    np.save(inputs_dense_dir / "queen_prior.npy", queen_prior.toarray())

    np.save(inputs_fixed_dir / "regression_design.npy", regression_design)

    # Save consistent reference outputs used by run.py debug checks.
    theta_external_true = np.array([tau_iid_true, tau_dense_true, prec_o_true])
    theta_internal_true = np.log(theta_external_true)
    x_true = np.concatenate([u_iid, u_dense, beta_true])

    np.save(ref_dir / "theta_internal.npy", theta_internal_true)
    np.save(ref_dir / "theta_external.npy", theta_external_true)
    np.save(ref_dir / "x.npy", x_true)

    q_prior = sparse.block_diag(
        [
            tau_iid_true * iid_prior,
            tau_dense_true * queen_prior,
            0.001 * sparse.identity(n_fixed),
        ],
        format="csc",
    )

    a_full = sparse.hstack(
        [iid_design, sparse.csc_matrix(queen_design), sparse.csc_matrix(regression_design)], format="csc"
    )
    q_cond = q_prior + prec_o_true * (a_full.T @ a_full)

    sparse.save_npz(ref_dir / "qprior.npz", q_prior)
    sparse.save_npz(ref_dir / "qcond.npz", q_cond)

    print("Generated stable synthetic bee_genomics data.")
    print(f"n_obs={n_obs}, n_fixed={n_fixed}, n_iid={n_iid}, n_dense={n_dense}")
    print(f"Saved observations to {BASE_DIR / 'observations.npy'}")


if __name__ == "__main__":
    main()
