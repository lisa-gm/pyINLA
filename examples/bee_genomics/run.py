import os
import sys

import numpy as np

from pathlib import Path

from dalia import xp, sp
from dalia.configs import dalia_config, likelihood_config, submodels_config
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import GenericSubModel, RegressionSubModel
from dalia.utils import (
    print_msg,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from examples_utils.parser_utils import parse_args  # noqa: E402

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(__file__).resolve().parent / "synthetic_data"

if __name__ == "__main__":
    print_msg("--- Example: Bee Genomics  ---")
    # consists of 2 generic submodels (iid + dense component) and 1 regression submodel (covariates + covariates) with a Gaussian likelihood

    # Check for parsed parameters
    args = parse_args()

    # Configurations of the generic submodel
    generic_dict_iid = {
        "type": "generic",
        "input_dir": f"{BASE_DIR}/inputs_iid",
        "tau": 4,  # has to be positive
        "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 1e-1},
    }
    generic_iid = GenericSubModel(
        config=submodels_config.parse_config(generic_dict_iid),
    )

    # Configurations of the generic submodel
    generic_dict_queenGRMinv = {
        "type": "generic",
        "input_dir": f"{BASE_DIR}/inputs_queenGRMinv",
        "ph_tau": {"type": "gamma", "alpha": 1.0, "beta": 1e-1},
    }
    generic_queenGRMinv = GenericSubModel(
        config=submodels_config.parse_config(generic_dict_queenGRMinv),
    )

    # fixed effects + intercept
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_fixed_effects",
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    # Likelihood
    likelihood_dict = {
        "type": "gaussian",
        "prior_hyperparameters": {"type": "gamma", "alpha": 1.0, "beta": 1e-1},
    }
    # Creation of the first model by combining the Generic submodel and the likelihood
    model = Model(
        submodels=[generic_iid, generic_queenGRMinv, regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    # Configurations of DALIA
    dalia_dict = {
        "solver": {"type": "dense"},
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    results = dalia.run()

    # load theta reference and set theta_internal to reference values
    theta_ref_internal = xp.load(f"{BASE_DIR}/reference_outputs/theta_internal.npy")
    theta_ref_external = xp.load(f"{BASE_DIR}/reference_outputs/theta_external.npy")
    x_ref = xp.load(f"{BASE_DIR}/reference_outputs/x.npy")

    print_msg("\n--- Results ---")
    print_msg("Theta values external:\n", results["theta"])
    print_msg("Theta values internal:\n", results["theta_internal"])
    print_msg("Internal Covariance of theta:\n", results["cov_theta_internal"])

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    print_msg("Theta reference external:\n", theta_ref_external)
    print_msg("Theta values external:\n", results["theta"])
    print_msg(
        "Norm (theta external - theta_ref_external):        ",
        f"{xp.linalg.norm(results['theta'] - theta_ref_external):.4e}",
    )
    print_msg(
        "Norm (theta internal - theta_ref_internal):        ",
        f"{xp.linalg.norm(results['theta_internal'] - theta_ref_internal):.4e}",
    )

    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref)/Norm(x_ref):                ",
        f"{xp.linalg.norm(results['x'] - x_ref) / xp.linalg.norm(x_ref):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional)
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    print_msg("\n--- Marginal distributions of the hyperparameters ---")
    marginals_hp = dalia.marginal_distributions_hp()

    prec_obs = marginals_hp["hyperparameters"]["prec_o"]
    quantile_pairs = prec_obs["quantiles"]["external"]["pairs"]

    print("Quantile pairs of prec_o:")
    for p, q in quantile_pairs:
        print(f"   {p:.3f} quantile: {q:.4f}")

    no_samples = 5000
    samples = dalia.sample_posterior_latent_parameters(n_samples=no_samples)

    n_iid_latent = generic_iid.n_latent_parameters

    iid_indices = xp.arange(n_iid_latent)  # indices of the iid latent parameters
    # compute generic indices corresponding to the same latent variables
    # NOTE: check that the iid submodel is actually first in the model, then generic & that they have the same number
    if generic_queenGRMinv.n_latent_parameters != n_iid_latent:
        raise ValueError(
            "The number of latent parameters in the generic submodel should be the same as in the iid submodel for this."
        )
    generic_indices = iid_indices + n_iid_latent

    # extract subset of relevant samples from different latent components
    mean_latent_idd = results["x"][iid_indices]
    mean_latent_generic = results["x"][generic_indices]
    
    samples_iid = samples[iid_indices, :]
    samples_generic = samples[generic_indices, :]
    
    print_msg("theta_keys: ", marginals_hp["hyperparameters"].keys())
    
    est_var_tau_iid = 1 / results['marginals_hp']['hyperparameters']['tau']['mean_external']
    print_msg(f"\nVariance of latent iid:        {xp.var(mean_latent_idd):.4e}")
    print_msg(f"Variance of samples (iid):     {np.mean(xp.var(samples_iid, axis=0)):.4e}")
    print_msg(f"Est. variance from iid hp:     {est_var_tau_iid:.4e}")
    print_msg(f"ref variance of iid hp:        {1 / theta_ref_external[0]:.4e}")
    
    est_var_tau_generic = 1 / results['marginals_hp']['hyperparameters']['tau_2']['mean_external']
    print_msg(f"\nVariance of latent generic:      {xp.var(mean_latent_generic):.4e}")
    print_msg(f"Variance of samples (generic):   {np.mean(xp.var(samples_generic, axis=0)):.4e}")
    print_msg(f"Est. variance from generic hp:   {est_var_tau_generic:.4e}")
    print_msg(f"ref variance of generic hp:      {1 / theta_ref_external[1]:.4e}")
    
    print_msg("\n--- Finished ---")
