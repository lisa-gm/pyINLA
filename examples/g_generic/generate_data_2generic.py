## generate synthetic data for a model with two different generic submodels
## One dense component, i.e. q = spd dense matrix, 
## and one iid component, i.e. q = identity matrix
## and some fixed effects

import os
import numpy as np
import sys

import numpy as np
import scipy.sparse as sp

sys.path.append("..")

np.random.seed(443)

path = os.path.dirname(__file__)

if __name__ == "__main__":
    n_dense_component = 500
    n_iid_component = n_dense_component
    n_fe = 4  # number of fixed effects
    n_latent_parameters = n_iid_component + n_dense_component + n_fe
    # increase number of observations per variable
    # technically per n_dense_component, as they refer to same "individual"
    n_obs_per_param = 2
    n_obs = n_dense_component * n_obs_per_param
    nrep = 1  # number of replicates

    q_temp = np.random.randn(n_dense_component, n_dense_component)
    q_dense = q_temp @ q_temp.T
    # normalize to not have arbitrarily large values in the precision matrix
    q_dense = 1 / np.max(q_dense) * q_dense

    tau_iid = 1.5
    tau_dense = 3.5
    s = 0.2  # sd
    theta_ref = [tau_iid, tau_dense, 1 / s**2]

    ## create iid generic component with diagonal precision matrix
    L_Q_prior_iid = np.sqrt(tau_iid) * np.eye(n_iid_component)
    z_iid = np.random.normal(size=n_iid_component)
    x_iid = np.linalg.solve(L_Q_prior_iid.T, z_iid)

    # create generic component with dense precision matrix
    Q_scaled = tau_dense * q_dense
    L_Q_prior = np.linalg.cholesky(Q_scaled)
    z = np.random.normal(size=n_dense_component)
    x_dense = np.linalg.solve(L_Q_prior.T, z)    
    
    # fixed effects 
    x_fe = np.random.uniform(low=-3, high=3, size=n_fe)

    # construct projection matrix
    # multiple observation per latent parameter
    a_temp = sp.kron(sp.eye(n_iid_component), np.ones((n_obs_per_param, 1)))
    a_iid = sp.kron(np.ones((nrep, 1)), a_temp)
    #a_dense = sp.kron(np.ones((nrep, 1)), sp.random(n_dense_component, n_dense_component, density=0.05))
    a_dense = sp.kron(np.ones((nrep, 1)), a_temp)
    a_fe = sp.csr_matrix(np.kron(np.ones((nrep, 1)), np.random.uniform(low=-1, high=1, size=(n_obs, n_fe))))
    a = sp.hstack([a_dense, a_iid, a_fe])
    x = np.concatenate([x_dense, x_iid, x_fe])

    # y = Ax + noise
    y = (a @ x) + np.random.normal(scale=s, size=a.shape[0] * nrep)

    # save the synthetic data
    path_files = f"{path}/inputs_2generic"
    os.makedirs(path_files, exist_ok=True)
    np.save(f"{path_files}/y.npy", y)
    
    # create subfolder for iid component
    os.makedirs(f"{path_files}/inputs_generic_iid", exist_ok=True)
    sp.save_npz(f"{path_files}/inputs_generic_iid/q.npz", sp.eye(n_iid_component))
    sp.save_npz(f"{path_files}/inputs_generic_iid/a.npz", a_iid)
    
    # create a subfolder for dense component
    os.makedirs(f"{path_files}/inputs_generic_dense", exist_ok=True)
    q_dense = sp.coo_matrix(q_dense)
    sp.save_npz(f"{path_files}/inputs_generic_dense/q.npz", q_dense)
    # save a as .npz
    sp.save_npz(f"{path_files}/inputs_generic_dense/a.npz", a_dense)
    # sparse.save_npz(f"{path}/inputs_generic/a.npz", a)
    
    # create a subfolder for fixed effects
    os.makedirs(f"{path_files}/inputs_fixed_effects", exist_ok=True)
    sp.save_npz(f"{path_files}/inputs_fixed_effects/a.npz", a_fe)

    # reference outputs
    os.makedirs(f"{path_files}/reference_outputs", exist_ok=True)
    np.save(f"{path_files}/reference_outputs/x_ref.npy", x)
    np.save(f"{path_files}/reference_outputs/theta_ref.npy", theta_ref)
