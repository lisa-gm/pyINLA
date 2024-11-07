# Copyright 2024 pyINLA authors. All rights reserved.

from pathlib import Path

import numpy as np

# import pytest
from conftest import pyinla_config_model

# from pyinla import ArrayLike
from scipy import sparse
from scipy.sparse import triu  # csc_matrix,

# from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.models.spatio_temporal import SpatioTemporalModel
from pyinla.utils.theta_utils import read_sym_CSC, theta_array2dict  # theta_dict2array

# have 2 folders with theta vector, matrices & and expected output


def test_spatio_temporal_Q_prior(
    pyinla_config_model,
):
    # load theta
    # load number of latent parameters
    n_latent_parameters = np.load("./tests/tests_models/inputs/n_latent_parameters.npy")
    theta = np.load("./tests/tests_models/inputs/theta.npy")

    # instantiate the model
    model_instance = SpatioTemporalModel(pyinla_config_model, n_latent_parameters)

    # call construct Q prior
    Q_prior = model_instance.construct_Q_prior(theta)

    # compare to expected output
    Q_prior_ref = np.load("./inputs/Q_prior.npz")

    assert np.allclose(Q_prior, Q_prior_ref)


def test_spatio_temporal_Q_conditional(
    pyinla_config_model,
):
    # instantiate the model

    # load hessian likelihood

    # call construct Q conditional

    # compare to expected output

    pass


# until tests are working use main to run the tests
if __name__ == "__main__":
    pyinla_config_model = pyinla_config_model()

    pyinla_config_model.simulation_dir = Path.cwd()
    # pyinla_config_model.input_dir = Path.cwd() / "inputs"
    pyinla_config_model.model.n_fixed_effects = 6

    n_latent_parameters = np.load("./inputs/n_latent_parameters.npy")
    theta = np.load("./inputs/theta.npy")

    model_instance = SpatioTemporalModel(pyinla_config_model, n_latent_parameters)
    theta_model, theta_likelihood = theta_array2dict(
        theta, model_instance.get_theta(), {}
    )
    # print(theta_model)

    # call construct Q prior
    Q_prior = model_instance.construct_Q_prior(theta_model)
    # print("Q prior PyINLA: \n", Q_prior[:5, :5].toarray())

    # compare to expected output
    # Q_prior_ref = np.load( Path.cwd() / "inputs" / "Q_prior.npz")
    Q_prior_ref = read_sym_CSC(Path.cwd() / "inputs" / "Q_prior_INLA_DIST_466_466.dat")
    # symmetrize matrix
    Q_prior_ref = Q_prior_ref + triu(Q_prior_ref.T, 1)
    # print("Q prior INLA DIST: \n", Q_prior_ref[:5, :5].toarray())

    if not np.allclose(Q_prior.toarray(), Q_prior_ref.toarray()):
        raise AssertionError("Q prior not the same!!")

    # call construct Q conditional
    a = sparse.load_npz(Path.cwd() / "inputs" / "a.npz")
    neg_hessian_likelihood_vec = np.loadtxt(
        Path.cwd() / "inputs" / "hess_eta_4600_1.dat"
    )
    # hessian likelihood is a vector, create sparse diagonal matrix from it
    hessian_likelihood = -1 * sparse.diags(neg_hessian_likelihood_vec)

    Q_conditional = model_instance.construct_Q_conditional(
        Q_prior, a, hessian_likelihood
    )
    # print("Q conditional PyINLA: \n", Q_conditional[:5, :5].toarray())

    Q_conditional_ref = read_sym_CSC(
        Path.cwd() / "inputs" / "Q_conditional_INLA_DIST_466_466.dat"
    )
    # symmetrize matrix
    Q_conditional_ref = Q_conditional_ref + triu(Q_conditional_ref.T, 1)
    # print("Q conditional INLA DIST: \n", Q_conditional_ref[:5, :5].toarray())

    if not np.allclose(Q_conditional.toarray(), Q_conditional_ref.toarray()):
        raise AssertionError("Q conditional not the same!!")

    print("Tests passed!! Q prior and Q conditional are the same as in INLA_DIST.")
