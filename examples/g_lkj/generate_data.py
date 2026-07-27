## Generate synthetic data for the LKJ submodel with multiple replicates
## For each replicate i:
##   x_i ~ N(0, Σ) where Σ = [σ₁²      ρσ₁σ₂]
##                            [ρσ₁σ₂     σ₂² ]
##   y_i = A_i x_i + ε_i, ε_i ~ N(0, σ²_ε I)

import os
import numpy as np
import scipy.sparse as sp

np.random.seed(402)

path = os.path.dirname(__file__)

if __name__ == "__main__":
    # Number of observations per replicate
    n_obs = 10
    n_replicates = 500  # number of replicates

    # True hyperparameters (external space) - shared across all replicates
    sigma1_true = 1.5
    sigma2_true = 0.8
    rho_true = -0.3  # correlation
    sigma_eps_true = 0.1  # observation noise
    prec_obs = 1.0 / (sigma_eps_true**2)  # precision of observation noise

    # Construct true covariance matrix Σ for 2D latent
    cov_matrix = np.array(
        [
            [sigma1_true**2, rho_true * sigma1_true * sigma2_true],
            [rho_true * sigma1_true * sigma2_true, sigma2_true**2],
        ]
    )
    L_cov = np.linalg.cholesky(cov_matrix)

    # Storage for all x_ref (concatenated across replicates)
    x_ref_all = np.zeros(n_replicates * 2)

    # Construct observation matrix A -> resuse same covariates for all replicates
    a_dense = np.hstack([np.ones((n_obs, 1)), np.random.randn(n_obs, 1)])
    a = sp.csr_matrix(a_dense)

    # Generate each replicate
    for i in range(n_replicates):
        np.random.seed(402 + i)  # different seed for each replicate

        # Sample latent parameters from the prior
        # x ~ N(0, Σ)
        z = np.random.normal(size=2)
        x_true = L_cov @ z

        # Generate observations: y = Ax + ε
        y = (a @ x_true) + np.random.normal(scale=sigma_eps_true, size=n_obs)

        # Save the synthetic data in replicate-specific folder
        input_dir = f"{path}/inputs_nrep{n_replicates}/replicate_{i+1}"
        os.makedirs(input_dir, exist_ok=True)

        # Save y
        np.save(f"{input_dir}/y.npy", y)

        # Save observation matrix in inputs folder
        os.makedirs(f"{input_dir}/inputs_lkj", exist_ok=True)
        sp.save_npz(f"{input_dir}/inputs_lkj/a.npz", a)

        # Store x_ref for later concatenation
        x_ref_all[i * 2 : (i + 1) * 2] = x_true

        print(f"Replicate {i+1}/{n_replicates}:")
        print(f"  x_true = {x_true}")
        print(f"  y first 5: {y[:5]}")

    # Save reference outputs (same for all replicates, hence in main folder)
    os.makedirs(f"{path}/inputs_nrep{n_replicates}/reference_outputs", exist_ok=True)

    # Save all x_ref concatenated
    np.save(f"{path}/inputs_nrep{n_replicates}/reference_outputs/x_ref.npy", x_ref_all)

    # Save true hyperparameters (theta in external space)
    # Order: [sigma1, sigma2, rho, prec_obs] - same for all replicates
    theta_ref = np.array([sigma1_true, sigma2_true, rho_true, prec_obs])
    np.save(f"{path}/inputs_nrep{n_replicates}/reference_outputs/theta_ref.npy", theta_ref)

    print(f"\nGenerated synthetic LKJ data with {n_replicates} replicates:")
    print(f"  n_obs per replicate = {n_obs}")
    print(f"  n_replicates = {n_replicates}")
    print(
        f"  sigma1_true = {sigma1_true}, sigma2_true = {sigma2_true}, rho_true = {rho_true}"
    )
    print(f"  sigma_eps_true = {sigma_eps_true}")
    print(f"  prec_obs_true = {prec_obs}")
    print(f"  Covariance matrix:\n{cov_matrix}")
    
# compute sample covariance matrix 
sample_cov_matrix = np.cov(x_ref_all.reshape(n_replicates, 2).T)
print(f"  Sample covariance matrix:\n{sample_cov_matrix}")    

