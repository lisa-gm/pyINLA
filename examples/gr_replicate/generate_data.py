# generate synthetic regression dataset


import os

# add .. to the path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

sys.path.append("..")

np.random.seed(41)

path = os.path.dirname(__file__)

if __name__ == "__main__":
    n_observations = 20
    n_latent_parameters = 6
    n_replicates = 3  # number of replicates

    prior_precision = 1e-3
    Sigma_prior = 1 / prior_precision * np.eye(n_latent_parameters)
    L_Sigma_prior = np.linalg.cholesky(Sigma_prior)

    theta_observations = 2.0
    print(f"theta_observations: {theta_observations}")
    theta_likelihood: dict = {"theta_observations": theta_observations}

    # generate x from a gaussian distribution of dimensions n_latent_parameters with mean 0 and precision exp(theta_observations)
    variance = 1 / theta_observations

    x_ref = np.zeros(
        n_replicates * n_latent_parameters
    )  # to store all x_ref for all replicates

    # Sample multiple latent parameters from the prior --- write everything in a loop
    # this makes writing to file easier
    for i in range(n_replicates):
        np.random.seed(41 + i)  # change the seed for each replicate
        z = np.random.normal(size=(n_latent_parameters,))

        x = L_Sigma_prior @ z
        a = sparse.random(n_observations, n_latent_parameters, density=0.5)

        eta = a @ x
        y = np.random.normal(eta, scale=np.sqrt(variance), size=n_observations)
        print(f"x: {x}")
        # print(f"y: {y}")
        # print(f"a: {a}")

        a = csr_matrix(a)
        print("A: \n", a[:10, :n_latent_parameters].toarray())
        print("y: ", y[:10])

        input_dir = f"{path}/inputs/replicate_{i+1}"
        os.makedirs(input_dir, exist_ok=True)

        # save the synthetic data
        np.save(f"{input_dir}/y.npy", y)
        # save a as .npz
        os.makedirs(f"{input_dir}/inputs_regression", exist_ok=True)
        sparse.save_npz(f"{input_dir}/inputs_regression/a.npz", a)

        # accumulate reference x
        x_ref[i * n_latent_parameters : (i + 1) * n_latent_parameters] = x

    # save original hyperparameter theta
    os.makedirs(f"{path}/reference_outputs", exist_ok=True)
    np.save(f"{path}/reference_outputs/theta_ref.npy", theta_observations)
    np.save(f"{path}/reference_outputs/x_ref.npy", x_ref)
