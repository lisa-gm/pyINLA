import os
from pathlib import Path

import numpy as np
from scipy import sparse


BASE_DIR = Path(__file__).resolve().parent / "synthetic_data"


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
    n_fixed = 50
    n_iid = 40
    n_dense = 60

    tau_iid_true = 3.0
    tau_dense_true = 10.0
    prec_o_true = 50.0

    # Directories expected by run.py
    inputs_iid_dir = BASE_DIR / "inputs_iid"
    inputs_dense_dir = BASE_DIR / "inputs_queenGRMinv"
    inputs_fixed_dir = BASE_DIR / "inputs_fixed_effects"
    ref_dir = BASE_DIR / "reference_outputs"

    for p in [inputs_iid_dir, inputs_dense_dir, inputs_fixed_dir, ref_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # 1) IID generic component
    group_idx = make_balanced_group_indices(n_obs, n_iid)
    rows = np.arange(n_obs)
    cols = group_idx
    data = np.ones(n_obs, dtype=float)
    a_iid = sparse.coo_matrix((data, (rows, cols)), shape=(n_obs, n_iid)).tocsc()
    q_iid = sparse.identity(n_iid, format="csc")

    # True latent effects
    u_iid = rng.normal(loc=0.0, scale=np.sqrt(1.0 / tau_iid_true), size=n_iid)

    # 2) Dense generic component: random design and SPD precision matrix.
    a_dense = rng.normal(loc=0.0, scale=1.0 / np.sqrt(n_dense), size=(n_obs, n_dense))

    m = rng.normal(loc=0.0, scale=0.2, size=(n_dense, n_dense))
    q_dense_np = m.T @ m + np.eye(n_dense)
    q_dense = sparse.csc_matrix(q_dense_np)

    cov_dense = np.linalg.inv(tau_dense_true * q_dense_np)
    l_dense = np.linalg.cholesky(cov_dense)
    u_dense = l_dense @ rng.normal(size=n_dense)

    # 3) Regression component: intercept + near-orthonormal covariates.
    x_raw = rng.normal(size=(n_obs, n_fixed - 1))
    q_cov, _ = np.linalg.qr(x_raw, mode="reduced")
    a_fixed = np.column_stack([np.ones(n_obs), q_cov])

    # Sparse signal setting: only a few fixed effects are nonzero.
    beta_true = np.zeros(n_fixed)
    beta_true[0] = 0.5
    n_nonzero = 8
    nonzero_idx = rng.choice(np.arange(1, n_fixed), size=n_nonzero, replace=False)
    beta_true[nonzero_idx] = rng.normal(loc=0.0, scale=0.2, size=n_nonzero)

    eta = a_iid @ u_iid + a_dense @ u_dense + a_fixed @ beta_true
    y = eta + rng.normal(loc=0.0, scale=np.sqrt(1.0 / prec_o_true), size=n_obs)

    # Save run.py inputs
    np.save(BASE_DIR / "y.npy", y)

    sparse.save_npz(inputs_iid_dir / "a.npz", a_iid)
    sparse.save_npz(inputs_iid_dir / "q.npz", q_iid)

    np.save(inputs_dense_dir / "a.npy", a_dense)
    sparse.save_npz(inputs_dense_dir / "q.npz", q_dense)

    np.save(inputs_fixed_dir / "a.npy", a_fixed)

    # Save consistent reference outputs used by run.py debug checks.
    theta_external_true = np.array([tau_iid_true, tau_dense_true, prec_o_true])
    theta_internal_true = np.log(theta_external_true)
    x_true = np.concatenate([u_iid, u_dense, beta_true])

    np.save(ref_dir / "theta_internal.npy", theta_internal_true)
    np.save(ref_dir / "x.npy", x_true)

    q_prior = sparse.block_diag(
        [
            tau_iid_true * q_iid,
            tau_dense_true * q_dense,
            0.001 * sparse.identity(n_fixed),
        ],
        format="csc",
    )

    a_full = sparse.hstack(
        [a_iid, sparse.csc_matrix(a_dense), sparse.csc_matrix(a_fixed)], format="csc"
    )
    q_cond = q_prior + prec_o_true * (a_full.T @ a_full)

    sparse.save_npz(ref_dir / "qprior.npz", q_prior)
    sparse.save_npz(ref_dir / "qcond.npz", q_cond)

    print("Generated stable synthetic bee_genomics data.")
    print(f"n_obs={n_obs}, n_fixed={n_fixed}, n_iid={n_iid}, n_dense={n_dense}")
    print(f"Saved y to {BASE_DIR / 'y.npy'}")


if __name__ == "__main__":
    main()
