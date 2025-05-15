# Copyright 2024-2025 pyINLA authors. All rights reserved.

import os
from abc import ABC
from pathlib import Path
from tabulate import tabulate

import numpy as np

from pyinla import ArrayLike, NDArray, sp, xp
from pyinla.configs.likelihood_config import LikelihoodConfig
from pyinla.configs.priorhyperparameters_config import (
    BetaPriorHyperparametersConfig,
    GaussianMVNPriorHyperparametersConfig,
    GaussianPriorHyperparametersConfig,
    PenalizedComplexityPriorHyperparametersConfig,
)
from pyinla.core.likelihood import Likelihood
from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.core.submodel import SubModel
from pyinla.likelihoods import BinomialLikelihood, GaussianLikelihood, PoissonLikelihood
from pyinla.prior_hyperparameters import (
    BetaPriorHyperparameters,
    GaussianMVNPriorHyperparameters,
    GaussianPriorHyperparameters,
    PenalizedComplexityPriorHyperparameters,
)
from pyinla.submodels import (
    BrainiacSubModel,
    RegressionSubModel,
    SpatialSubModel,
    SpatioTemporalSubModel,
)
from pyinla.utils import scaled_logit, add_str_header, boxify


class Model(ABC):
    """Core class for statistical models."""

    def __init__(
        self,
        submodels: list[SubModel],
        likelihood_config: LikelihoodConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""
        self.modeltype = kwargs.get("modeltype", "Default Model")

        # Check the order of the submodels, we want the SpatioTemporalSubModel first
        # as this will decide the sparsity pattern of the precision matrix.
        for i, submodel in enumerate(submodels):
            if isinstance(submodel, SpatioTemporalSubModel):
                submodels.insert(0, submodels.pop(i))

        self.submodels: list[SubModel] = submodels

        self.n_fixed_effects: int = 0

        # For each submodel...
        theta: ArrayLike = []
        theta_keys: ArrayLike = []
        self.hyperparameters_idx: ArrayLike = [0]
        self.prior_hyperparameters: list[PriorHyperparameters] = []

        for submodel in self.submodels:
            # ...initialize their prior hyperparameters matching their hyperparameters
            if isinstance(submodel, SpatioTemporalSubModel):
                # Spatial hyperparameters
                if isinstance(submodel.config.ph_s, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_s,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_s, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_s,
                            hyperparameter_type="r_s",
                        )
                    )

                # Temporal hyperparameters
                if isinstance(submodel.config.ph_t, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_t,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_t, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_t,
                            hyperparameter_type="r_t",
                        )
                    )

                # Sigma spatio-temporal hyperparameters
                if isinstance(
                    submodel.config.ph_st, GaussianPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_st,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_st, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_st,
                            hyperparameter_type="sigma_st",
                        )
                    )
            elif isinstance(submodel, SpatialSubModel):
                # spatial range
                if isinstance(submodel.config.ph_s, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_s,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_s, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_s,
                            hyperparameter_type="r_s",
                        )
                    )

                # spatial variation
                if isinstance(submodel.config.ph_e, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            config=submodel.config.ph_e,
                        )
                    )
                elif isinstance(
                    submodel.config.ph_e, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            config=submodel.config.ph_e,
                            hyperparameter_type="sigma_e",
                        )
                    )
            elif isinstance(submodel, RegressionSubModel):
                self.n_fixed_effects += submodel.n_fixed_effects

            elif isinstance(submodel, BrainiacSubModel):
                # h2 hyperparameters
                if isinstance(submodel.config.ph_h2, BetaPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        BetaPriorHyperparameters(
                            config=submodel.config.ph_h2,
                        )
                    )

                # alpha hyperparameters
                if isinstance(
                    submodel.config.ph_alpha, GaussianMVNPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        GaussianMVNPriorHyperparameters(
                            config=submodel.config.ph_alpha,
                        )
                    )
            else:
                raise ValueError("Unknown submodel type")

            # ...and read their hyperparameters
            theta_submodel, theta_keys_submodel = submodel.config.read_hyperparameters()

            theta.append(theta_submodel)
            theta_keys += theta_keys_submodel

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_submodel)
            )

        # Add the likelihood hyperparameters
        (
            lh_hyperparameters,
            lh_hyperparameters_keys,
        ) = likelihood_config.read_hyperparameters()

        theta.append(lh_hyperparameters)
        self.theta: NDArray = xp.concatenate(theta)

        theta_keys += lh_hyperparameters_keys
        self.theta_keys: NDArray = theta_keys

        self.n_hyperparameters = self.theta.size

        # --- Initialize the latent parameters and the design matrix
        self.n_latent_parameters: int = 0
        self.latent_parameters_idx: list[int] = [0]

        for submodel in self.submodels:
            self.n_latent_parameters += submodel.n_latent_parameters
            self.latent_parameters_idx.append(self.n_latent_parameters)

        self.x: NDArray = xp.zeros(self.n_latent_parameters)

        data = []
        rows = []
        cols = []
        for i, submodel in enumerate(self.submodels):
            # Convert csc_matrix to coo_matrix to allow slicing
            coo_submodel_a = submodel.a.tocoo()
            data.append(coo_submodel_a.data)
            rows.append(coo_submodel_a.row)
            cols.append(
                coo_submodel_a.col
                + self.latent_parameters_idx[i]
                * xp.ones(coo_submodel_a.col.size, dtype=int)
            )

            self.x[
                self.latent_parameters_idx[i] : self.latent_parameters_idx[i + 1]
            ] = submodel.x_initial

        self.a: sp.sparse.spmatrix = sp.sparse.coo_matrix(
            (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
            shape=(submodel.a.shape[0], self.n_latent_parameters),
        )

        # TODO: not so efficient ...
        self.permutation_latent_variables = xp.arange(self.n_latent_parameters)
        self.inverse_permutation_latent_variables = xp.arange(self.n_latent_parameters)

        # --- Load observation vector
        input_dir = Path(
            kwargs.get("input_dir", os.path.dirname(submodels[0].config.input_dir))
        )

        y: NDArray = np.load(input_dir / "y.npy")
        if xp == np:
            self.y: NDArray = y
        else:
            self.y: NDArray = xp.asarray(y)

        self.y = self.y.flatten()

        self.n_observations: int = self.y.shape[0]

        # --- Initialize likelihood
        # TODO: clean this -> so that for brainiac model we don't add additional hyperperameter
        if likelihood_config.type == "gaussian":
            self.likelihood: Likelihood = GaussianLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )

            if self.submodels[0] == BrainiacSubModel:
                # skip setting prior as it's already set in the submodel
                print(
                    "Brainiac model detected. Skipping setting prior hyperparameters as already set."
                )
            # Instantiate the prior hyperparameters for the likelihood
            elif isinstance(
                likelihood_config.prior_hyperparameters,
                GaussianPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        config=likelihood_config.prior_hyperparameters,
                    )
                )
            elif isinstance(
                likelihood_config.prior_hyperparameters,
                PenalizedComplexityPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparameters(
                        config=likelihood_config.prior_hyperparameters,
                        hyperparameter_type="prec_o",
                    )
                )
        elif likelihood_config.type == "poisson":
            self.likelihood: Likelihood = PoissonLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )
        elif likelihood_config.type == "binomial":
            self.likelihood: Likelihood = BinomialLikelihood(
                n_observations=self.n_observations,
                config=likelihood_config,
            )

        self.likelihood_config: LikelihoodConfig = likelihood_config

        # --- Recurrent variables
        self.Q_prior = None
        self.Q_prior_data_mapping = [0]
        self.Q_conditional = None
        self.Q_conditional_data_mapping = [0]

    def construct_Q_prior(self) -> sp.sparse.spmatrix:
        kwargs = {}

        if self.Q_prior is None:
            # During the first construction of Q_prior, we allocate the memory for
            # the data and the mapping of each submodel's to the Q prior matrix.
            rows = []
            cols = []
            data = []

            for i, submodel in enumerate(self.submodels):
                if isinstance(submodel, SpatioTemporalSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])
                elif isinstance(submodel, SpatialSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])
                elif isinstance(submodel, BrainiacSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])
                elif isinstance(submodel, RegressionSubModel):
                    ...

                submodel_Q_prior = submodel.construct_Q_prior(**kwargs)

                rows.append(
                    submodel_Q_prior.row
                    + self.latent_parameters_idx[i] * xp.ones(len(submodel_Q_prior.row))
                )
                cols.append(
                    submodel_Q_prior.col
                    + self.latent_parameters_idx[i] * xp.ones(len(submodel_Q_prior.col))
                )
                data.append(submodel_Q_prior.data)

                self.Q_prior_data_mapping.append(
                    self.Q_prior_data_mapping[i] + len(submodel_Q_prior.data)
                )

            self.Q_prior: sp.sparse.csc_matrix = sp.sparse.csc_matrix(
                (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
                shape=(self.n_latent_parameters, self.n_latent_parameters),
            )

        else:
            for i, submodel in enumerate(self.submodels):
                if isinstance(submodel, RegressionSubModel):
                    ...
                elif isinstance(submodel, SpatioTemporalSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])
                elif isinstance(submodel, SpatialSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])
                elif isinstance(submodel, BrainiacSubModel):
                    for hp_idx in range(
                        self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
                    ):
                        kwargs[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])

                submodel_Q_prior = submodel.construct_Q_prior(**kwargs)

                self.Q_prior.data[
                    self.Q_prior_data_mapping[i] : self.Q_prior_data_mapping[i + 1]
                ] = submodel_Q_prior.data

        return self.Q_prior

    def construct_Q_conditional(
        self,
        eta: NDArray,
    ) -> float:
        """Construct the conditional precision matrix.

        Note
        ----
        Input of the hessian of the likelihood is a diagonal matrix.
        The negative hessian is required, therefore the minus in front.

        """

        # TODO: need to vectorize
        # hessian_likelihood_diag = hessian_diag_finite_difference_5pt(
        #     self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
        # )
        # hessian_likelihood = diags(hessian_likelihood_diag)

        if self.likelihood_config.type == "gaussian":
            kwargs = {
                "eta": eta,
                "theta": float(self.theta[-1]),
            }
        else:
            kwargs = {
                "eta": eta,
            }

        if isinstance(self.submodels[0], BrainiacSubModel):
            # Brainiac specific rule
            kwargs["h2"] = float(self.theta[0])
            d_matrix = self.submodels[0].evaluate_d_matrix(**kwargs)
        else:
            # General rules
            d_matrix = self.likelihood.evaluate_hessian_likelihood(**kwargs)

        self.Q_conditional = self.Q_prior - self.a.T @ d_matrix @ self.a

        return self.Q_conditional

    def construct_information_vector(
        self,
        eta: NDArray,
        x_i: NDArray,
    ) -> NDArray:
        """Construct the information vector."""

        if isinstance(self.submodels[0], BrainiacSubModel):
            kwargs = {"h2": float(self.theta[0])}
            gradient_likelihood = self.submodels[0].evaluate_gradient_likelihood(
                eta=eta, y=self.y, **kwargs
            )

        else:
            gradient_likelihood = self.likelihood.evaluate_gradient_likelihood(
                eta=eta,
                y=self.y,
                theta=self.theta[self.hyperparameters_idx[-1] :],
            )

        information_vector: NDArray = (
            -1 * self.Q_prior @ x_i + self.a.T @ gradient_likelihood
        )

        return information_vector

    def is_likelihood_gaussian(self) -> bool:
        """Check if the likelihood is Gaussian."""
        return self.likelihood_config.type == "gaussian"

    def evaluate_log_prior_hyperparameters(self) -> float:
        """Evaluate the log prior hyperparameters."""
        log_prior = 0.0

        # if BFGS and model scale differ: rescale -- generalize
        if isinstance(self.submodels[0], BrainiacSubModel):
            #
            theta_interpret = self.theta.copy()
            theta_interpret[0] = scaled_logit(self.theta[0], direction="backward")
            # TODO: multivariate prior for a ... need to generalize for now:
            log_prior += self.prior_hyperparameters[0].evaluate_log_prior(
                theta_interpret[0]
            )

            log_prior += self.prior_hyperparameters[1].evaluate_log_prior(
                theta_interpret[1:]
            )
        else:
            theta_interpret = self.theta

            for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
                log_prior += prior_hyperparameter.evaluate_log_prior(theta_interpret[i])

        return log_prior

    def get_theta_likelihood(self) -> NDArray:
        """Return the likelihood hyperparameters."""

        if isinstance(self.submodels[0], BrainiacSubModel):
            theta_likelihood = 1 - scaled_logit(self.theta[0], direction="backward")
        else:
            theta_likelihood = self.theta[self.hyperparameters_idx[-1] :]

        return theta_likelihood

    def evaluate_likelihood(self, eta: NDArray, **kwargs) -> float:
        """Evaluate the likelihood."""

        if isinstance(self.submodels[0], BrainiacSubModel):
            kwargs["h2"] = float(self.theta[0])
            likelihood = self.submodels[0].evaluate_likelihood(eta, self.y, **kwargs)
        else:
            likelihood = self.likelihood.evaluate_likelihood(
                eta, self.y, theta=self.theta[self.hyperparameters_idx[-1] :]
            )

        return likelihood

    def __str__(self) -> str:
        """String representation of the model."""
        str_representation = ""

        # --- Make the Model() table ---
        headers = ["Number of Hyperparameters", "Number of Latent Parameters", "Number of Observations", "Type of Likelihood"]
        values = [self.n_hyperparameters, self.n_latent_parameters, self.n_observations, self.likelihood_config.type.capitalize()]

        model_table = tabulate([headers, values], tablefmt="fancy_grid", colalign=("center", "center", "center", "center"))

        # Add the header title
        model_table = add_str_header("Default Model", model_table)

        # --- Add the submodel information ---
        # Create headers and values for the submodel table
        submodels_str_representation = []
        for submodel in self.submodels:
            submodels_str_representation.append(str(submodel))

        lines_list = [s.splitlines() for s in submodels_str_representation]
        max_len = max(len(lines) for lines in lines_list)

        # Pad each list of lines to the same length
        for lines in lines_list:
            lines += [''] * (max_len - len(lines))

        # Concatenate corresponding lines
        result_lines = ['  '.join(parts) for parts in zip(*lines_list)]
        submodel_jointed_representation = '\n'.join(result_lines)

        # Add the submodel header title
        submodel_jointed_representation = add_str_header("Submodels", submodel_jointed_representation)

        # Combine the model and submodel tables
        str_representation = model_table + "\n" + submodel_jointed_representation

        return boxify(str_representation)

    def get_solver_parameters(self) -> dict:
        """Get the solver parameters."""
        diagonal_blocksize = None
        n_diag_blocks = None
        arrowhead_blocksize = 0
        if isinstance(self.submodels[0], SpatioTemporalSubModel):
            diagonal_blocksize = self.submodels[0].ns
            n_diag_blocks = self.submodels[0].nt

        for i in range(1, len(self.submodels)):
            if isinstance(self.submodels[i], RegressionSubModel):
                arrowhead_blocksize += self.submodels[i].n_latent_parameters

        param = {
            "diagonal_blocksize": diagonal_blocksize,
            "n_diag_blocks": n_diag_blocks,
            "arrowhead_blocksize": arrowhead_blocksize,
        }

        return param
    
    
    def construct_a_predict(self) -> sp.sparse.spmatrix:
        """Construct the design matrix for prediction."""
        
        data = []
        rows = []
        cols = []
                
        rows_a_predict = 0
        for i, submodel in enumerate(self.submodels):
            # Convert csc_matrix to coo_matrix to allow slicing
            coo_submodel_a_predict = submodel.load_a_predict().tocoo()
            data.append(coo_submodel_a_predict.data)
            rows.append(coo_submodel_a_predict.row)
            cols.append(
                coo_submodel_a_predict.col
                + self.latent_parameters_idx[i]
                * xp.ones(coo_submodel_a_predict.col.size, dtype=int)
            )
            
            # the number of rows in all of them is the same
            rows_a_predict = coo_submodel_a_predict.shape[0]
                    
        self.a_predict: sp.sparse.spmatrix = sp.sparse.coo_matrix(
            (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
            shape=(rows_a_predict, self.n_latent_parameters),
        )

    def total_number_fixed_effects(self) -> int:
        """Get the number of fixed effects."""
        return self.n_fixed_effects