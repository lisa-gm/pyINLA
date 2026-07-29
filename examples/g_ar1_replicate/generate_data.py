import os
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse import csc_matrix
from scipy.linalg import cholesky

BASE_DIR: Path = Path(__file__).parent

if __name__ == "__main__":

    np.random.seed(359)
    n = 100
    n_replicates = 2  # number of replicates
    replicate_intercept = True  # if True, each replicate gets its own intercept latent

    ## define priors
    phi = 0.7
    s2 = 5
    tau = 1 / s2
    # noise obs
    obs_noise_prec = 10
    theta_original = [
        phi,
        tau,
        obs_noise_prec,
    ]

    denom = s2 * (1 - phi**2)

    diag = [(1 + phi**2) / denom] * n
    diag[0] = diag[-1] = 1 / denom
    off_diag = [-phi / denom] * (n - 1)

    Q = sp.diags([diag, off_diag, off_diag], [0, -1, 1])

    # Compute sparse Cholesky factorization: Q = L @ L.T
    Q_csc = Q.tocsc()

    # Method 1: Use dense Cholesky (for moderate sizes this is still efficient)
    Q_dense = Q.toarray()
    L_dense = cholesky(Q_dense, lower=True)
    L = csc_matrix(L_dense)

    # True intercept is always singular; replicate_intercept controls model flexibility
    intercept = np.random.uniform(-5, 5)
    intercepts = np.full(n_replicates if replicate_intercept else 1, intercept)

    x_ref_ar1 = np.zeros(n_replicates * n)
    y_ref = np.zeros(n_replicates * n)

    # Store A once
    a_ar1 = sp.eye(n, format="csr")
    # Regression design matrix: when intercept is shared, replicate across all obs.
    if replicate_intercept:
        a_regression = sp.csr_matrix(np.ones((n, 1)))
    else:
        a_regression = sp.csr_matrix(np.ones((n_replicates * n, 1)))

    # Generate data for each replicate
    for rep in range(n_replicates):
        np.random.seed(5 + rep)  # Different seed for each replicate

        # Sample latent AR1 process for this replicate
        z = np.random.normal(0, 1, size=n)
        u = spsolve_triangular(L, z, lower=True)

        print(f"\n--- Replicate {rep + 1} ---")
        print("u: ", u[:10])

        intercept = intercepts[rep] if replicate_intercept else intercepts[0]
        eta = u + intercept
        noise = np.random.normal(0, np.sqrt(1 / obs_noise_prec), size=eta.shape)
        y = eta + noise

        print("y: ", y[:10])

        # Store concatenated AR1 latent states and observations
        x_ref_ar1[rep * n : (rep + 1) * n] = u
        y_ref[rep * n : (rep + 1) * n] = y

    # Final x_ref layout matches model latent ordering.
    # shared-intercept mode: [all AR1 states, one intercept]
    # replicated-intercept mode: [all AR1 states, intercept per replicate]
    x_ref = np.concatenate((x_ref_ar1, intercepts))

    # Save consolidated dataset under inputs_nrep*
    output_dir = BASE_DIR / f"inputs_nrep{n_replicates}"
    os.makedirs(output_dir / "inputs_ar1", exist_ok=True)
    os.makedirs(output_dir / "inputs_regression", exist_ok=True)
    os.makedirs(output_dir / "reference_outputs", exist_ok=True)

    np.save(output_dir / "y.npy", y_ref)
    sp.save_npz(output_dir / "inputs_ar1" / "a.npz", a_ar1)
    sp.save_npz(output_dir / "inputs_regression" / "a.npz", a_regression)

    np.save(output_dir / "reference_outputs" / "x_ref.npy", x_ref)
    np.save(output_dir / "reference_outputs" / "theta_ref.npy", theta_original)

    print("\n--- Summary ---")
    print(f"Generated {n_replicates} replicates with {n} observations each")
    print(f"AR1 latent dimension total: {n_replicates * n}")
    if replicate_intercept:
        print(f"Replicated intercept latent dimension total: {n_replicates}")
    else:
        print("Shared intercept latent dimension: 1")
