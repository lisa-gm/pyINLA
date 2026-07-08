## Generate synthetic data for the LKJ submodel
## x ~ N(0, Σ) where Σ = [σ₁²      ρσ₁σ₂]
##                        [ρσ₁σ₂     σ₂² ]
## y = Ax + ε, ε ~ N(0, σ²_ε)

import os
import numpy as np
import scipy.sparse as sp

np.random.seed(4002)

path = os.path.dirname(__file__)

if __name__ == "__main__":
    # Number of observations
    n_obs = 1000

    # True hyperparameters (external space)
    sigma1_true = 1.5
    sigma2_true = 0.8
    rho_true = 0.6  # correlation
    sigma_eps_true = 0.1  # observation noise
    prec_obs = 1.0 / (sigma_eps_true**2)  # precision of observation noise

    # Construct true covariance matrix Σ for 2D latent
    sigma_eps_var = sigma_eps_true**2
    cov_matrix = np.array(
        [
            [sigma1_true**2, rho_true * sigma1_true * sigma2_true],
            [rho_true * sigma1_true * sigma2_true, sigma2_true**2],
        ]
    )

    # Sample latent parameters from the prior
    # x ~ N(0, Σ)
    L_cov = np.linalg.cholesky(cov_matrix)
    z = np.random.normal(size=2)
    x_true = L_cov @ z

    # Construct observation matrix A
    # A is n_obs × 2, can be sparse or dense
    # Simple design: some random entries, some structure
    a_dense = np.random.randn(n_obs, 2) * 0.5 + np.ones((n_obs, 2))
    a = sp.csr_matrix(a_dense)

    # Generate observations: y = Ax + ε
    y = (a @ x_true) + np.random.normal(scale=sigma_eps_true, size=n_obs)

    # Save the synthetic data
    os.makedirs(f"{path}/inputs_lkj", exist_ok=True)
    os.makedirs(f"{path}/reference_outputs", exist_ok=True)

    # Save y at top level
    np.save(f"{path}/y.npy", y)
    # Save observation matrix in inputs folder
    sp.save_npz(f"{path}/inputs_lkj/a.npz", a)

    # Save true latent parameters
    np.save(f"{path}/reference_outputs/x_ref.npy", x_true)

    # Save true hyperparameters (theta in external space)
    # Order: [sigma1, sigma2, rho, prec_obs]
    theta_ref = np.array([sigma1_true, sigma2_true, rho_true, prec_obs])
    np.save(f"{path}/reference_outputs/theta_ref.npy", theta_ref)

    print(f"Generated synthetic LKJ data:")
    print(f"  n_obs = {n_obs}")
    print(f"  x_true = {x_true}")
    print(
        f"  sigma1_true = {sigma1_true}, sigma2_true = {sigma2_true}, rho_true = {rho_true}"
    )
    print(f"  sigma_eps_true = {sigma_eps_true}")
    print(f"  Covariance matrix:\n{cov_matrix}")
