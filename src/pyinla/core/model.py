# Copyright 2024 pyINLA authors. All rights reserved.

from abc import ABC

import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from pyinla import ArrayLike, xp, sp

from pyinla.core.pyinla_config import PyinlaConfig
from pyinla.core.submodel import SubModel
from pyinla.submodels.regression import RegressionModel
from pyinla.submodels.spatio_temporal import SpatioTemporalModel

from pyinla.likelihoods import BinomialLikelihood, GaussianLikelihood, PoissonLikelihood

from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.prior_hyperparameters import (
    GaussianPriorHyperparameters,
    PenalizedComplexityPriorHyperparameters,
)

from pyinla.core.pyinla_config import (
    RegressionSubModelConfig,
    SpatioTemporalSubModelConfig,
    GaussianPriorHyperparametersConfig,
    PenalizedComplexityPriorHyperparametersConfig,
)


class Model(ABC):
    """Core class for statistical models."""

    def __init__(
        self,
        pyinla_config: PyinlaConfig,
    ) -> None:
        """Initializes the model."""
        self.pyinla_config: PyinlaConfig = pyinla_config

        # --- Initialize the submodels and their prior hyperparameters
        # Initialization order priviledges the spatio-temporal submodel (first)
        # and all others (second).
        self.submodels: list[SubModel] = []
        self.prior_hyperparameters: list[PriorHyperparameters] = []
        submodel_to_instanciate = [xp.arrange(len(pyinla_config.model.submodels))]
        for i, submodel_config in enumerate(self.pyinla_config.model.submodels):
            if isinstance(submodel_config, SpatioTemporalSubModelConfig):
                # Instancitate the SpatialTemporalModel
                self.submodels.append(
                    SpatioTemporalModel(submodel_config, pyinla_config.input_dir)
                )
                submodel_to_instanciate.pop(i)

                # Instantiate the prior hyperparameters
                # ... for the spatial range
                if isinstance(submodel_config.ph_s, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            ph_config=submodel_config.ph_s, hyperparameter_type="r_s"
                        )
                    )
                elif isinstance(
                    submodel_config.ph_s, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            ph_config=submodel_config.ph_s, hyperparameter_type="r_s"
                        )
                    )

                # ... for the temporal range
                if isinstance(submodel_config.ph_t, GaussianPriorHyperparametersConfig):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            ph_config=submodel_config.ph_t, hyperparameter_type="r_t"
                        )
                    )
                elif isinstance(
                    submodel_config.ph_t, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            ph_config=submodel_config.ph_t, hyperparameter_type="r_t"
                        )
                    )

                # ... for the spatio-temporal variation
                if isinstance(
                    submodel_config.ph_st, GaussianPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        GaussianPriorHyperparameters(
                            ph_config=submodel_config.ph_st,
                            hyperparameter_type="sigma_st",
                        )
                    )
                elif isinstance(
                    submodel_config.ph_st, PenalizedComplexityPriorHyperparametersConfig
                ):
                    self.prior_hyperparameters.append(
                        PenalizedComplexityPriorHyperparameters(
                            ph_config=submodel_config.ph_st,
                            hyperparameter_type="sigma_st",
                        )
                    )

        for i in submodel_to_instanciate:
            if isinstance(submodel_config[i], RegressionSubModelConfig):
                self.submodels.append(
                    RegressionModel(submodel_config, pyinla_config.input_dir)
                )
            else:
                raise ValueError("Unknown submodel type.")

        # --- Initialize the hyperparameters array
        theta = []
        theta_keys = []
        self.hyperparameters_idx = [0]

        for submodel_config in self.pyinla_config.model.submodels:
            theta_submodel, theta_keys_submodel = submodel_config.read_hyperparameters()

            theta.append(theta_submodel)
            theta_keys.append(theta_keys_submodel)

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_submodel)
            )

        lh_hyperparameters, lh_hyperparameters_keys = (
            self.pyinla_config.model.likelihood.read_hyperparameters()
        )

        theta.append(lh_hyperparameters)
        theta_keys.append(lh_hyperparameters_keys)

        self.theta = xp.concatenate(theta)
        self.theta_keys = xp.concatenate(theta_keys)

        self.n_hyperparameters = len(self.theta)

        # --- Initialize the latent parameters and the design matrix
        self.n_latent_parameters = 0
        self.latent_parameters_idx = [0]

        for submodel in self.submodels:
            self.n_latent_parameters += submodel.n_latent_parameters
            self.latent_parameters_idx.append(self.n_latent_parameters)

        self.x: ArrayLike = xp.zeros(self.n_latent_parameters)

        data = []
        rows = []
        cols = []
        for i, submodel in enumerate(self.submodels):
            data.append(submodel.a.data)
            rows.append(submodel.a.row)
            cols.append(
                submodel.a.col
                + self.latent_parameters_idx[i] * xp.ones(submodel.a.col.size[0])
            )

            self.x[
                self.latent_parameters_idx[i] : self.latent_parameters_idx[i + 1]
            ] = submodel.x_initial

        self.a: spmatrix = sp.sparse.coo_matrix(
            (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
            shape=(submodel.a.shape[0], self.n_latent_parameters),
        )

        # --- Load observation vector
        y = np.load(pyinla_config.input_dir / "y.npy")
        if xp == np:
            self.y = y
        else:
            self.y = xp.asarray(y)

        self.n_observations = self.y.shape[0]

        # --- Initialize likelihood
        if self.pyinla_config.model.likelihood.type == "gaussian":
            self.likelihood = GaussianLikelihood(pyinla_config, self.n_observations)

            # Instantiate the prior hyperparameters for the likelihood
            if isinstance(
                self.pyinla_config.model.likelihood.prior_hyperparameters,
                GaussianPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        ph_config=self.pyinla_config.model.likelihood.prior_hyperparameters,
                        hyperparameter_type="prec_o",
                    )
                )
            elif isinstance(
                self.pyinla_config.model.likelihood.prior_hyperparameters,
                PenalizedComplexityPriorHyperparametersConfig,
            ):
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparametersConfig(
                        ph_config=self.pyinla_config.model.likelihood.prior_hyperparameters,
                        hyperparameter_type="prec_o",
                    )
                )
        elif self.pyinla_config.model.likelihood.type == "poisson":
            self.likelihood = PoissonLikelihood(pyinla_config, self.n_observations)
        elif self.pyinla_config.model.likelihood.type == "binomial":
            self.likelihood = BinomialLikelihood(pyinla_config, self.n_observations)

        # --- Recurrent variables
        self.Q_prior = None
        self.Q_prior_data_mapping = [0]
        self.Q_conditional = None
        self.Q_conditional_data_mapping = [0]

    def construct_Q_prior(self) -> spmatrix:

        if self.Q_prior is None:
            rows = []
            cols = []
            data = []

            for i, submodel in enumerate(self.submodels):

                kwargs = {}
                if isinstance(submodel, RegressionModel):
                    ...
                elif isinstance(submodel, SpatioTemporalModel):
                    kwargs = {
                        "theta": self.theta[
                            self.hyperparameters_idx[i] : self.hyperparameters_idx[
                                i + 1
                            ]
                        ],
                        "theta_keys": self.theta_keys[
                            self.hyperparameters_idx[i] : self.hyperparameters_idx[
                                i + 1
                            ]
                        ],
                    }
                submodel_Q_prior = submodel.construct_Q_prior(kwargs=kwargs)

                rows.append(
                    submodel_Q_prior.row
                    + self.latent_parameters_idx[i]
                    * xp.ones(submodel_Q_prior.row.size[0])
                )
                cols.append(
                    submodel_Q_prior.col
                    + self.latent_parameters_idx[i]
                    * xp.ones(submodel_Q_prior.col.size[0])
                )
                data.append(submodel_Q_prior.data)

                self.Q_prior_data_mapping.append(
                    self.Q_prior_data_mapping[i] + len(submodel_Q_prior.data)
                )

            self.Q_prior: spmatrix = coo_matrix(
                (xp.concatenate(data), (xp.concatenate(rows), xp.concatenate(cols))),
                shape=(self.n_latent_parameters, self.n_latent_parameters),
            )

        else:

            for i, submodel in enumerate(self.submodels):
                kwargs = {}
                if isinstance(submodel, RegressionModel):
                    ...
                elif isinstance(submodel, SpatioTemporalModel):
                    kwargs = {
                        "theta": self.theta[
                            self.hyperparameters_idx[i] : self.hyperparameters_idx[
                                i + 1
                            ]
                        ],
                        "theta_keys": self.theta_keys[
                            self.hyperparameters_idx[i] : self.hyperparameters_idx[
                                i + 1
                            ]
                        ],
                    }
                submodel_Q_prior = submodel.construct_Q_prior(kwargs=kwargs)

                self.Q_prior.data[
                    self.Q_prior_data_mapping[i] : self.Q_prior_data_mapping[i + 1]
                ] = submodel_Q_prior.data

        return self.Q_prior

    def construct_Q_conditional(
        self,
        eta: ArrayLike,
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
        hessian_likelihood = self.likelihood.evaluate_hessian_likelihood(
            kwargs={
                "eta": eta,
                "theta": self.theta[self.hyperparameters_idx[-1] :],
            }
        )

        self.Q_conditional = self.Q_prior - self.a.T @ hessian_likelihood @ self.a

        return self.Q_conditional

    def construct_information_vector(
        self,
        eta: ArrayLike,
        x_i: ArrayLike,
    ) -> ArrayLike:
        """Construct the information vector."""

        # TODO: need to vectorize !!
        # gradient_likelihood = gradient_finite_difference_5pt(
        #     self.likelihood.evaluate_likelihood, eta, self.y, theta_likelihood
        # )
        gradient_likelihood = self.likelihood.evaluate_gradient_likelihood(
            eta, self.y, kwargs={"theta": self.theta[self.hyperparameters_idx[-1] :]}
        )

        information_vector = -1 * self.Q_prior @ x_i + self.a.T @ gradient_likelihood

        return information_vector

    def evaluate_log_prior_hyperparameters(self) -> float:
        """Evaluate the log prior hyperparameters."""
        log_prior = 0.0

        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            log_prior += prior_hyperparameter.evaluate_log_prior(self.theta[i])

        return log_prior
