## Generate synthetic data for the LKJ submodel with multiple groups
## Each group has shared underlying random effects with LKJ covariance
## x = [u_1, v_1, u_2, v_2, ..., u_G, v_G, beta_0]
## where [u_g, v_g] ~ N(0, Σ) for each group g
## Σ = [σ₁²      ρσ₁σ₂]
##     [ρσ₁σ₂     σ₂² ]
## y = Ax + ε, ε ~ N(0, σ²_ε)

import os
import numpy as np
import scipy.sparse as sp

np.random.seed(4002)

path = os.path.dirname(__file__)

if __name__ == "__main__":
    # Number of groups and observations
    n_groups = 5
    n_obs_per_group = 25
    n_obs = n_groups * n_obs_per_group

    # True fixed effects
    beta_0_true = -1.1   # global slope

    # True hyperparameters for random effects (external space)
    sigma1_true = 0.8  # variance of random intercept
    sigma2_true = 0.2  # variance of random slope
    rho_true = -0.85  # correlation between random intercept and slope
    sigma_eps_true = 0.2  # observation noise
    prec_obs = 1.0 / (sigma_eps_true**2)  # precision of observation noise

    # Construct true covariance matrix Σ for random effects (intercept, slope)
    cov_matrix = np.array(
        [
            [sigma1_true**2, rho_true * sigma1_true * sigma2_true],
            [rho_true * sigma1_true * sigma2_true, sigma2_true**2],
        ]
    )

    # Sample latent parameters and construct observation matrix A
    # x = [u_1, v_1, u_2, v_2, ..., u_G, v_G, beta_0]
    x_true = np.zeros(2 * n_groups + 1)
    
    # Pre-allocate design matrix
    n_cols = 1 + 2 * n_groups
    a_dense = np.zeros((n_obs, n_cols))
    
    # Set every even column from 0 onwards to 1s (group-specific intercepts)
    for col_idx in range(0, 2 * n_groups, 2):
        g = col_idx // 2
        start_idx = g * n_obs_per_group
        end_idx = (g + 1) * n_obs_per_group
        a_dense[start_idx:end_idx, col_idx] = 1.0
    
    # Sample random effects and fill odd columns with group-specific slopes
    L_cov = np.linalg.cholesky(cov_matrix)
    for g in range(n_groups):
        # Sample random effects from LKJ covariance
        z = np.random.normal(size=2)
        random_effects = L_cov @ z
        x_true[2*g : 2*(g+1)] = random_effects
        
        # Fill odd column (group-specific slope)
        col_idx = 2 * g + 1
        start_idx = g * n_obs_per_group
        end_idx = (g + 1) * n_obs_per_group
        a_dense[start_idx:end_idx, col_idx] = np.random.normal(0, 1, n_obs_per_group)
        
    # assume global mean to be zero
    # Global slope at column 2*n_groups+1 (random predictor)
    a_dense[:, 2 * n_groups] = np.random.normal(0, 1, n_obs)
    
    # Fixed effect at the end of x
    x_true[2*n_groups] = beta_0_true
    
    a = sp.csr_matrix(a_dense)

    # Generate observations: y = Ax + ε
    y = (a @ x_true) + np.random.normal(scale=sigma_eps_true, size=n_obs)

    # Save the synthetic data
    output_dir = f"{path}/inputs_ngroups{n_groups}"
    os.makedirs(f"{output_dir}/inputs_lkj", exist_ok=True)
    os.makedirs(f"{output_dir}/inputs_regression", exist_ok=True)
    os.makedirs(f"{output_dir}/reference_outputs", exist_ok=True)

    # Save y at top level
    np.save(f"{output_dir}/y.npy", y)
    
    # Split design matrix: LKJ gets group-specific effects, Regression gets global effects
    a_lkj = sp.csr_matrix(a_dense[:, :2*n_groups])  # Columns 0 to 2*n_groups-1
    a_regression = sp.csr_matrix(a_dense[:, 2*n_groups:])  # Columns 2*n_groups onward
    
    # Save split observation matrices
    sp.save_npz(f"{output_dir}/inputs_lkj/a.npz", a_lkj)
    sp.save_npz(f"{output_dir}/inputs_regression/a.npz", a_regression)

    # Save true latent parameters
    np.save(f"{output_dir}/reference_outputs/x_ref.npy", x_true)

    # Save true hyperparameters (theta in external space)
    # Order: [sigma1, sigma2, rho, prec_obs]
    theta_ref = np.array([sigma1_true, sigma2_true, rho_true, prec_obs])
    np.save(f"{output_dir}/reference_outputs/theta_ref.npy", theta_ref)

    print(f"Generated synthetic LKJ data (random slope model with {n_groups} groups):")
    print(f"  Output directory: {output_dir}")
    print(f"  n_groups = {n_groups}, n_obs_per_group = {n_obs_per_group}, total n_obs = {n_obs}")
    print(f"  Fixed effects: beta_0 = {beta_0_true}")
    print(f"  x_true shape: {x_true.shape}")
    print(f"  x_true (last 6 values): {x_true[-6:]}")
    print(f"  Design matrix a shape: {a_dense.shape}")
    print(
        f"  sigma1_true (intercept var) = {sigma1_true}, sigma2_true (slope var) = {sigma2_true}, rho_true = {rho_true}"
    )
    print(f"  sigma_eps_true = {sigma_eps_true}")
    print(f"  Covariance matrix:\n{cov_matrix}")

    # ============================================================================
    # Generate federated data: each group becomes a site with local a and y
    # ============================================================================
    federated_output_dir = f"{path}/ngroups{n_groups}_federated"
    
    # Create shared reference outputs folder (same as centralized)
    os.makedirs(f"{federated_output_dir}/reference_outputs", exist_ok=True)
    np.save(f"{federated_output_dir}/reference_outputs/x_ref.npy", x_true)
    np.save(f"{federated_output_dir}/reference_outputs/theta_ref.npy", theta_ref)
    
    # Create site-specific folders
    for g in range(n_groups):
        start_idx = g * n_obs_per_group
        end_idx = (g + 1) * n_obs_per_group
        
        site_dir = f"{federated_output_dir}/site{g+1}"
        os.makedirs(f"{site_dir}/inputs_lkj", exist_ok=True)
        os.makedirs(f"{site_dir}/inputs_regression", exist_ok=True)
        
        # Site-specific observations
        y_site = y[start_idx:end_idx]
        np.save(f"{site_dir}/y.npy", y_site)
        
        # Site-specific design matrix for LKJ: only 2 columns (u_g, v_g)
        # This is just the g-th pair of columns from the full LKJ matrix
        a_lkj_site = sp.csr_matrix(a_dense[start_idx:end_idx, 2*g:2*g+2])
        sp.save_npz(f"{site_dir}/inputs_lkj/a.npz", a_lkj_site)
        
        # Site-specific design matrix for regression: 1 column (global slope)
        a_regression_site = sp.csr_matrix(a_dense[start_idx:end_idx, 2*n_groups:])
        sp.save_npz(f"{site_dir}/inputs_regression/a.npz", a_regression_site)
    
    print(f"\nGenerated federated data:")
    print(f"  Output directory: {federated_output_dir}")
    print(f"  Number of sites: {n_groups}")
    print(f"  Each site contains: site / (inputs_lkj/a.npz, inputs_regression/a.npz, y.npy)")
