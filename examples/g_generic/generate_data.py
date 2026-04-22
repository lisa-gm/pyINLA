## generate synthetic data for the generic submodel

import os
import numpy as np

# add .. to the path
import sys

import numpy as np
import scipy.sparse as sp

sys.path.append("..")

np.random.seed(44)

path = os.path.dirname(__file__)

if __name__ == "__main__":
    n_latent_parameters = 300
    nrep = 1

    q_temp = np.random.randn(n_latent_parameters, n_latent_parameters)
    q = q_temp @ q_temp.T

    tau = 2.5
    s = 0.1  # sd, prec 100
    theta_ref = [tau, 1 / s**2]

    Q_scaled = tau * q

    L_Q_prior = np.linalg.cholesky(Q_scaled)
    z = np.random.normal(size=n_latent_parameters)
    x = np.linalg.solve(L_Q_prior.T, z)

    # construct projection matrix
    a = sp.kron(np.ones((nrep, 1)), sp.eye(n_latent_parameters))

    # y = Ax + noise
    y = (a @ x) + np.random.normal(scale=s, size=n_latent_parameters * nrep)

    # save the synthetic data
    np.save(f"{path}/y.npy", y)

    # create a subfolder called inputs
    os.makedirs(f"{path}/inputs_generic", exist_ok=True)
    q = sp.coo_matrix(q)
    sp.save_npz(f"{path}/inputs_generic/q.npz", q)

    # save a as .npz
    sp.save_npz(f"{path}/inputs_generic/a.npz", a)
    # sparse.save_npz(f"{path}/inputs_generic/a.npz", a)

    # save original latent parameters
    os.makedirs(f"{path}/reference_outputs", exist_ok=True)

    np.save(f"{path}/reference_outputs/x_ref.npy", x)

    # save original hyperparameter theta
    np.save(f"{path}/reference_outputs/theta_ref.npy", theta_ref)
