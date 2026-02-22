## generate synthetic data for the generic submodel

import os
import numpy as np

# add .. to the path
import sys

import numpy as np
import scipy.sparse as sp

sys.path.append("..")

np.random.seed(41)

path = os.path.dirname(__file__)

if __name__ == "__main__":
    n_observations = 500
    n_latent_parameters = 6

    # model: y = A @ x + epsilon, with x ~ N(0, 1/tau*Cmatrix) and epsilon ~ N(0, 1/theta_observations*I)

    tau = 0.8
    prec_noise = 5.0
    theta_ref = [tau, prec_noise]

    # generate random positive definite matrix for the covariance of x
    random_matrix = np.random.rand(n_latent_parameters, n_latent_parameters)
    q = random_matrix @ random_matrix.T + 1e-3 * np.eye(
        n_latent_parameters
    )  # make it positive definite
    # print(f"q: \n{q}")
    q = sp.coo_matrix(q)

    Q_prior = tau * q

    # sample x from the prior distribution
    L_Q_prior = np.linalg.cholesky(Q_prior.toarray())
    z = np.random.normal(size=n_latent_parameters)
    x = np.linalg.solve(L_Q_prior, z)

    a = sp.random(n_observations, n_latent_parameters, density=0.5)
    # print(f"A: \n{a.toarray()}")

    y = a @ x + np.random.normal(scale=1 / prec_noise, size=n_observations)
    print(f"x: {x}")
    # print(f"y: {y}")

    # save the synthetic data
    np.save(f"{path}/y.npy", y)

    # create a subfolder called inputs
    os.makedirs(f"{path}/inputs_generic", exist_ok=True)
    sp.save_npz(f"{path}/inputs_generic/q.npz", q)

    # save a as .npz
    sp.save_npz(f"{path}/inputs_generic/a.npz", a)
    # sparse.save_npz(f"{path}/inputs_generic/a.npz", a)

    # save original latent parameters
    os.makedirs(f"{path}/reference_outputs", exist_ok=True)

    np.save(f"{path}/reference_outputs/x_ref.npy", x)

    # save original hyperparameter theta
    np.save(f"{path}/reference_outputs/theta_ref.npy", theta_ref)
