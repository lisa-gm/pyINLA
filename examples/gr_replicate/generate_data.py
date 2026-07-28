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
    n_observations = 100
    n_latent_parameters = 6
    n_replicates = 10  # number of replicates

    prior_precision = 1e-3
    Sigma_prior = 1 / prior_precision * np.eye(n_latent_parameters)
    L_Sigma_prior = np.linalg.cholesky(Sigma_prior)

    theta_observations = 2.0
    print(f"theta_observations: {theta_observations}")

    # generate x from a gaussian distribution of dimensions n_latent_parameters with mean 0 and precision exp(theta_observations)
    variance = 1 / theta_observations

    # one shared A matrix for all replicates
    a = sparse.random(n_observations, n_latent_parameters, density=0.5)
    a = csr_matrix(a)

    # store all replicates consecutively
    x_ref = np.zeros(n_replicates * n_latent_parameters)
    y_ref = np.zeros(n_replicates * n_observations)

    print("A: \n", a[:10, :n_latent_parameters].toarray())

    # Sample multiple latent parameters from the prior
    for i in range(n_replicates):
        np.random.seed(41 + i)  # change the seed for each replicate
        z = np.random.normal(size=(n_latent_parameters,))

        x = L_Sigma_prior @ z

        eta = a @ x
        y = np.random.normal(eta, scale=np.sqrt(variance), size=n_observations)
        print(f"x: {x}")
        print("y: ", y[:10])

        # accumulate reference x
        x_ref[i * n_latent_parameters : (i + 1) * n_latent_parameters] = x
        y_ref[i * n_observations : (i + 1) * n_observations] = y

    # save consolidated dataset under inputs_nrep*
    output_dir = f"{path}/inputs_nrep{n_replicates}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/inputs_regression", exist_ok=True)
    os.makedirs(f"{output_dir}/reference_outputs", exist_ok=True)

    sparse.save_npz(f"{output_dir}/inputs_regression/a.npz", a)
    np.save(f"{output_dir}/y.npy", y_ref)
    np.save(f"{output_dir}/reference_outputs/x_ref.npy", x_ref)
    np.save(f"{output_dir}/reference_outputs/theta_ref.npy", theta_observations)
