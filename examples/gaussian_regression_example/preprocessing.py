# generate synthetic regression dataset


import os

# add .. to the path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

sys.path.append("..")

from save_to_file import write_sparse_csc_matrix

# set seed
np.random.seed(41)

path = os.path.dirname(__file__)

if __name__ == "__main__":
    n_observations = 1000
    n_latent_parameters = 6

    z = np.random.normal(size=n_latent_parameters)

    prior_precision = 1e-3
    Sigma_prior = 1 / prior_precision * np.eye(n_latent_parameters)
    L_Sigma_prior = np.linalg.cholesky(Sigma_prior)
    x = L_Sigma_prior @ z
    a = sparse.random(n_observations, n_latent_parameters, density=0.5)

    theta_observations = np.log(3)
    print(f"theta_observations: {theta_observations}")
    theta_likelihood: dict = {"theta_observations": theta_observations}

    # generate x from a gaussian distribution of dimensions n_latent_parameters with mean 0 and precision exp(theta_observations)
    variance = 1 / np.exp(theta_observations)
    eta = a @ x
    y = np.random.normal(eta, scale=np.sqrt(variance), size=n_observations)
    print(f"x: {x}")
    # print(f"y: {y}")
    # print(f"a: {a}")

    a = csr_matrix(a)
    print("A: \n", a[:10, :n_latent_parameters].toarray())
    print("y: ", y[:10])

    # create a subfolder called inputs
    os.makedirs(f"{path}/inputs", exist_ok=True)

    # save the synthetic data
    np.save(f"{path}/inputs/y.npy", y)
    # save a as .npz
    sparse.save_npz(f"{path}/inputs/a.npz", a)

    # save original latent parameters
    np.save(f"{path}/inputs/x_original.npy", x)

    # save original hyperparameter theta
    np.save(f"{path}/inputs/theta_original.npy", theta_observations)

    # save the synthetic data in INLA_DIST readable format
    os.makedirs(f"{path}/INLA_DIST_inputs", exist_ok=True)

    # write theta to file as a column vector
    dim_theta = 1
    # write number to file numpy
    np.savetxt(
        f"{path}/INLA_DIST_inputs/theta_original_{dim_theta}_1.dat",
        np.array([theta_observations]),
    )

    # write y to file as a column vector
    np.savetxt(f"{path}/INLA_DIST_inputs/y_{n_observations}_1.dat", y)

    # write a to file as a sparse matrix in CSC format
    write_sparse_csc_matrix(
        a, f"{path}/INLA_DIST_inputs/Ax_{n_observations}_{n_latent_parameters}.dat"
    )
