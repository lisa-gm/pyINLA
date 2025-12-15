from pathlib import Path
import numpy as np
import scipy.sparse as sp
import time

from dalia import xp
from dalia.core.dalia import DALIA
from dalia.core.model import Model
from dalia.submodels import BrainiacSubModel
from dalia.configs import dalia_config, likelihood_config, submodels_config

if __name__ == "__main__":
    print(f"Running BRAINIAC model on synthetic dataset.")

    path_inputs : Path = Path("inputs_brainiac")
    path_reference : Path = path_inputs / "reference"

    # 1. Load model parameters
    model_params = np.load(path_inputs / "model_params.npy", allow_pickle=True).item()
    n_observations = model_params["n_observations"]
    n_features = model_params["n_features"]
    n_annotations_per_features = model_params["n_annotations_per_features"]
    h2 = model_params["h2"]
    sigma_a2 = model_params["sigma_a2"]

    # 2. Load references
    theta_reference = np.load(path_reference / "theta.npy")
    x_reference = np.load(path_reference / "beta.npy")

    # 3. Create starting values for DALIA
    initial_h2 = theta_reference[0] - 0.1
    initial_alpha = theta_reference[1:] + 0.5 * np.random.randn(n_annotations_per_features - 1)


    # 4. Initialize the Brainiac submodel and the DALIA model
    brainiac = BrainiacSubModel(
        config=submodels_config.parse_config({
            "type": "brainiac",
            "input_dir": str(path_inputs.resolve()),
            "h2": initial_h2,
            "alpha": initial_alpha,
            "ph_h2": {"type": "beta", "alpha": 5.0, "beta": 1.0},
            "ph_alpha": {
                "type": "gaussian_mvn",
                "mean": theta_reference[1:],
                "precision": (1.0 / sigma_a2) * sp.eye(n_annotations_per_features),
            },
        })
    )
    model = Model(
        submodels=[brainiac],
        likelihood_config=likelihood_config.parse_config({"type": "gaussian", "fix_hyperparameters": True}),
    )
    print(model)

    # 5. Initialize DALIA
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config({
            "solver": {"type": "dense"},
            "minimize": {
                "max_iter": 50,
                "gtol": 1e-3,
                "disp": True,
            },
            "inner_iteration_max_iter": 50,
            "eps_inner_iteration": 1e-3,
            "eps_gradient_f": 1e-3,
            "simulation_dir": ".",
        })
    )

    # 6. Run inference
    tic = time.perf_counter()
    result = dalia.run()
    toc = time.perf_counter()
    print(f"DALIA finished in {toc - tic:.2f} seconds.")

    # 7. Compare to reference solution
    print("\n------ Compare to reference solution ------\n")
    print("theta_reference: ", theta_reference)
    print("theta dalia:", result["theta"])

    print("norm(x_reference - x) = ", np.linalg.norm(x_reference - result["x"]))

    # 8. Check marginal variances of latent parameters
    var_latent_params = result["marginal_variances_latent"]
    Q_conditional = dalia.model.construct_Q_conditional(eta=model.a @ model.x)

    if sp.issparse(Q_conditional):
        Q_inv_ref = xp.linalg.inv(Q_conditional.toarray())
    else:
        Q_inv_ref = xp.linalg.inv(Q_conditional)
    print(
        f"Norm (marginal variances of latent parameters - reference): {np.linalg.norm(var_latent_params - xp.diag(Q_inv_ref)):.4e}",
    )

    # 9. Compute marginal distributions of the hyperparameters
    marginals_hyperparameters = dalia.marginal_distributions_hp() 