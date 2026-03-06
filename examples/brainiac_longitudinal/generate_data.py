import os
import shutil

import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Clean old data folders
    folders_to_clean = ["inputs_brainiac", "inputs_trend", "inputs_subjects", "reference_outputs"]
    for folder in folders_to_clean:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Cleaned old data: {folder}")

    np.random.seed(5)

    # =============================================
    # Parameters
    # =============================================
    n_subjects    = 2000
    t_per_subject = 5
    n_obs         = n_subjects * t_per_subject
    b             = 12000   # genetic effect dimension
    m             = 2       # annotation dimension

    # =============================================
    # Data generation
    # =============================================
    subject_id = np.repeat(np.arange(n_subjects), t_per_subject)
    t = np.tile(np.arange(t_per_subject), n_subjects).astype(float)
    t = (t - t.mean()) / t.std()

    # Annotation matrix z: (b, m)
    z = np.random.rand(b, m)
    os.makedirs(os.path.join(base_dir, "inputs_brainiac"), exist_ok=True)
    np.save(os.path.join(base_dir, "inputs_brainiac", "z.npy"), z)

    # Design matrix A: (n_obs, b)
    A = np.random.randn(n_obs, b)
    A_sp = sp.csc_matrix(A)
    sp.save_npz(os.path.join(base_dir, "inputs_brainiac", "a.npz"), A_sp)

    # True hyperparameters
    h2       = 0.8
    sigma_a2 = 0.1
    alpha    = np.random.normal(0, np.sqrt(sigma_a2), (m, 1))
    theta_original = np.concatenate(([h2], alpha.flatten()))

    os.makedirs(os.path.join(base_dir, "reference_outputs"), exist_ok=True)
    np.save(os.path.join(base_dir, "reference_outputs", "theta_original.npy"), theta_original)

    # Prior precision matrix Q_prior = diag(1 / (h2 * Phi))
    exp_Z_alpha = np.exp(z @ alpha)
    Phi         = exp_Z_alpha / np.sum(exp_Z_alpha)
    h2_phi      = h2 * Phi.flatten()
    Qprior      = diags(1.0 / h2_phi)
    sp.save_npz(os.path.join(base_dir, "inputs_brainiac", "Qprior_original.npz"), Qprior)

    # Trend submodel design matrix: [intercept, t] -> (n_obs, 2)
    X_trend = np.column_stack([np.ones(n_obs), t])
    os.makedirs(os.path.join(base_dir, "inputs_trend"), exist_ok=True)
    sp.save_npz(os.path.join(base_dir, "inputs_trend", "a.npz"), sp.csc_matrix(X_trend))

    # Subject random-effects design matrix: (n_obs, n_subjects)
    X_subjects = np.zeros((n_obs, n_subjects))
    for i, s_id in enumerate(subject_id):
        X_subjects[i, s_id] = 1
    os.makedirs(os.path.join(base_dir, "inputs_subjects"), exist_ok=True)
    sp.save_npz(os.path.join(base_dir, "inputs_subjects", "a.npz"), sp.csc_matrix(X_subjects))

    # Sample latent variables
    beta_var = 1.0 / Qprior.diagonal()
    beta     = np.random.normal(0, np.sqrt(beta_var)).reshape(b, 1)
    np.save(os.path.join(base_dir, "reference_outputs", "beta_original.npy"), beta.flatten())

    sigma_u2   = 1.0
    u          = np.random.normal(0, np.sqrt(sigma_u2), n_subjects)
    u_obs      = u[subject_id].reshape(-1, 1)
    mu0, mu1   = 0.5, -0.3
    mean_trend = (mu0 + mu1 * t).reshape(-1, 1)
    sigma_eps2 = 1.0 - h2
    eps        = np.random.normal(0, np.sqrt(sigma_eps2), (n_obs, 1))

    # Observations: Y = mean_trend + u_obs + A @ beta + eps
    y = mean_trend + u_obs + A @ beta + eps
    np.save(os.path.join(base_dir, "y.npy"), y)

    print(f"y shape: {y.shape} saved to {base_dir}")
    print(f"Data generation complete: n_obs={n_obs}, b={b}, m={m}")
    print(f"z.shape = {z.shape}, A.shape = {A.shape}")
    print(f"z.shape[0] == A.shape[1]: {z.shape[0] == A.shape[1]}")
