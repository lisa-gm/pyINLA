import os
import sys

import numpy as np
import scipy.sparse as sp

from scipy.stats import multivariate_normal, poisson

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    n = 100

    ## define priors
    s2 = 0.05  # 0.7
    tau = 1 / s2
    phi = 0.5  # 0.9
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
    L = np.linalg.cholesky(Q.toarray())
    Cov = np.linalg.inv(Q.toarray())

    geom_mean = np.exp(np.mean(np.log(Cov.diagonal())))
    print("Geometric mean of Qinv diagonal: ", geom_mean)

    print(Q.toarray())
    print(np.linalg.inv(Q.toarray()))
    print(np.round(Q.toarray() @ np.linalg.inv(Q.toarray()), 6))
    # exit()

    mv = multivariate_normal(mean=np.zeros(n), cov=Cov, seed=3)

    intercept = 2
    u = mv.rvs()
    x = np.concatenate((u, [intercept]))
    print("x: ", x)
    np.save("reference_outputs/x_original.npy", x)
    x_initial = u + np.random.normal(0, 0.3, size=len(u))
    np.save("inputs_ar1/x.npy", u)
    np.save("reference_outputs/theta_original.npy", theta_original)

    a_ar1 = sp.eye(n)
    sp.save_npz("inputs_ar1/a.npz", a_ar1)

    a_regression = sp.csr_matrix(np.ones((n, 1)))
    sp.save_npz("inputs_regression/a.npz", a_regression)

    eta = a_ar1 @ u + intercept

    print("eta: ", eta)
    np.save("inputs_ar1/x_original.npy", eta)

    noise = np.random.normal(0, np.sqrt(1 / obs_noise_prec), size=eta.shape)
    print("noise: ", noise)
    y = eta + noise
    np.save("y.npy", y)

    print("y: ", y)

    Qprior = sp.block_diag([Q, sp.csr_matrix([[0.001]])])
    # print("Qprior : \n", Qprior.toarray())

    a = sp.hstack([a_ar1, a_regression])
    Qcond = Qprior + obs_noise_prec * a.T @ a
    print("Qcond: \n", Qcond.toarray())

    b = obs_noise_prec * a.T @ y
    # -xp.exp(theta) * (eta - y)
    # beta_initial + np.linalg.solve(
    #     Qconditional.toarray(), information_vector
    # )
    x_est = np.linalg.solve(Qcond.toarray(), b)
    print("x_est: ", x_est)

    print("eta est : ", a @ x_est)
    print("eta :     ", a @ x)
