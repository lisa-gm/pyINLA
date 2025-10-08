import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, spsolve_triangular
from scipy.sparse import csc_matrix
from scipy.linalg import cholesky

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    np.random.seed(5)
    n = 1000

    ## define priors
    s2 = 5  
    tau = 1 / s2
    phi = 0.9  
    # noise obs
    obs_noise_prec = 100
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
    # For tridiagonal matrix, we can use dense Cholesky on small blocks or scipy
    Q_csc = Q.tocsc()
    
    print("Q shape:", Q.shape, "Q nnz:", Q.nnz)
    print("Q sparsity:", 100 * Q.nnz / (Q.shape[0] * Q.shape[1]), "%")
    print(Q.toarray()[:6, :6])
    
    # Method 1: Use dense Cholesky (for moderate sizes this is still efficient)
    Q_dense = Q.toarray()
    L_dense = cholesky(Q_dense, lower=True)
    L = csc_matrix(L_dense)
    
    print("L nnz:", L.nnz, "L sparsity:", 100 * L.nnz / (L.shape[0] * L.shape[1]), "%")
    
    # Efficient sampling: generate z ~ N(0,I), then solve L @ u = z
    z = np.random.normal(0, 1, size=n)
    
    # Solve L @ u = z using sparse triangular solver
    u = spsolve_triangular(L, z, lower=True)
    
    # Verify the sampling worked correctly
    print("Sample u statistics - mean:", np.mean(u), "std:", np.std(u), ". Should be around sqrt(s2) =", np.sqrt(s2))
    
    intercept = 2

    x = np.concatenate((u, [intercept]))
    print("x: ", x[:10])
    
    np.save("reference_outputs/x_original.npy", x)
    np.save("inputs_ar1/x.npy", u)
    np.save("reference_outputs/theta_original.npy", theta_original)

    a_ar1 = sp.eye(n)
    sp.save_npz("inputs_ar1/a.npz", a_ar1)

    a_regression = sp.csr_matrix(np.ones((n, 1)))
    sp.save_npz("inputs_regression/a.npz", a_regression)

    eta = a_ar1 @ u + intercept

    print("eta: ", eta[:6])
    np.save("inputs_ar1/x_original.npy", eta)

    noise = np.random.normal(0, np.sqrt(1 / obs_noise_prec), size=eta.shape)
    print("noise: ", noise[:10])
    y = eta + noise
    np.save("y.npy", y)

    print("y: ", y[:10])

    Qprior = sp.block_diag([Q, sp.csr_matrix([[0.001]])])

    a = sp.hstack([a_ar1, a_regression]) # a_ar1 #
    Qcond = Qprior + obs_noise_prec * a.T @ a
    print("Qcond: \n", Qcond.toarray()[:6,:6])

    b = obs_noise_prec * a.T @ y
    print("b: ", b[:10])
    #x_est = np.linalg.solve(Qcond.toarray(), b)
    x_est = spsolve(csc_matrix(Qcond), b)
    print("norm(x - x_est): ", np.linalg.norm(x - x_est))

    print("norm(eta - eta_est): ", np.linalg.norm(a @ x - a @ x_est))
    print("normalized norm(eta - eta_est): ", np.linalg.norm(a @ x - a @ x_est) / np.linalg.norm(a @ x))