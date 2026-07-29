import os
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

    intercept = np.random.uniform(-5, 5)

    # Store reference x for all replicates
    x_ref = np.zeros(n_replicates * (n + 1))

    # Generate data for each replicate
    for rep in range(n_replicates):
        np.random.seed(5 + rep)  # Different seed for each replicate

        # Sample latent AR1 process for this replicate
        z = np.random.normal(0, 1, size=n)
        u = spsolve_triangular(L, z, lower=True)
        x = np.concatenate((u, [intercept]))

        print(f"\n--- Replicate {rep + 1} ---")
        print("x: ", x[:10])

        # Same latent structure for all replicates, but different observation noise
        a_ar1 = sp.eye(n)
        a_regression = sp.csr_matrix(np.ones((n, 1)))
        a = sp.hstack([a_ar1, a_regression])  # Combined observation matrix

        eta = a @ x
        noise = np.random.normal(0, np.sqrt(1 / obs_noise_prec), size=eta.shape)
        y = eta + noise

        print("y: ", y[:10])

        # Save replicate-specific data
        replicate_dir = BASE_DIR / "old_inputs" / f"replicate_{rep + 1}"
        os.makedirs(replicate_dir / "inputs_ar1", exist_ok=True)
        os.makedirs(replicate_dir / "inputs_regression", exist_ok=True)

        np.save(replicate_dir / "y.npy", y)
        sp.save_npz(replicate_dir / "inputs_ar1" / "a.npz", a_ar1)
        sp.save_npz(replicate_dir / "inputs_regression" / "a.npz", a_regression)

        # Store reference x
        x_ref[rep * (n + 1) : (rep + 1) * (n + 1)] = x

    # Save reference outputs (shared by all replicates)
    os.makedirs(replicate_dir / "reference_outputs", exist_ok=True)
    np.save(replicate_dir / "reference_outputs" / "x_ref.npy", x_ref)
    np.save(replicate_dir / "reference_outputs" / "theta_original.npy", theta_original)

    print("\n--- Summary ---")
    print(f"Generated {n_replicates} replicates with {n} observations each")
    print(f"Latent dimension per replicate: {n + 1} (n={n}, +1 for intercept)")
