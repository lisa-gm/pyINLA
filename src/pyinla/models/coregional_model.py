# Copyright 2024-2025 pyINLA authors. All rights reserved.

import re

import cupy as cp

import numpy as np
from scipy.sparse import spmatrix

from pyinla import ArrayLike, NDArray, sp, xp, backend_flags
from pyinla.configs.models_config import CoregionalModelConfig
from pyinla.core.model import Model
from pyinla.core.prior_hyperparameters import PriorHyperparameters
from pyinla.prior_hyperparameters import (
    GaussianPriorHyperparameters,
    PenalizedComplexityPriorHyperparameters,
)
from pyinla.submodels import RegressionSubModel, SpatialSubModel, SpatioTemporalSubModel
from pyinla.utils import bdiag_tiling, get_host

if backend_flags["cupy_avail"]:
    from cupyx.profiler import time_range
    import cupy as cp

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


class CoregionalModel(Model):
    """Core class for statistical models."""

    def __init__(
        self,
        models: list[Model],
        coregional_model_config: CoregionalModelConfig,
        **kwargs,
    ) -> None:
        """Initializes the model."""

        self.models: list[Model] = models

        # Check the coregionalization type (Spacial or SpatioTemporal)
        self.coregionalization_type: str
        self.n_models: int = coregional_model_config.n_models
        assert self.n_models == len(
            self.models
        ), "Number of models does not match the number of models in the CoregionalModelConfig"
        self.n_spatial_nodes: int = None
        self.n_temporal_nodes: int = 1
        self.n_fixed_effects_per_model: int = 0
        for i, model in enumerate(self.models):
            if i == 0:
                if isinstance(model.submodels[0], SpatioTemporalSubModel):
                    self.coregionalization_type = "spatio_temporal"
                    self.n_spatial_nodes = model.submodels[0].ns
                    self.n_temporal_nodes = model.submodels[0].nt
                elif isinstance(model.submodels[0], SpatialSubModel):
                    self.coregionalization_type = "spatial"
                    self.n_spatial_nodes = model.submodels[0].ns
                else:
                    print("model.submodels[0]: ", model.submodels[0])
                    raise ValueError(
                        "Invalid model type. Must be 'spatial' or 'spatio-temporal'."
                    )
                if len(model.submodels) > 1:
                    if isinstance(model.submodels[1], RegressionSubModel):
                        self.n_fixed_effects_per_model = model.submodels[
                            1
                        ].n_fixed_effects
            else:
                # Check that all models are the same
                if isinstance(model.submodels[0], SpatioTemporalSubModel):
                    if self.coregionalization_type != "spatio_temporal":
                        raise ValueError(
                            f"Model {model} is not of the same type as the first model (SpatioTemporalModel)"
                        )
                    # Check that the size of the SpatioTemporal fields are the same
                    if (
                        model.submodels[0].ns != self.n_spatial_nodes
                        or model.submodels[0].nt != self.n_temporal_nodes
                    ):
                        raise ValueError(
                            f"Model {model} is not of the same size as the first model (SpatioTemporalModel)"
                        )
                elif isinstance(model.submodels[0], SpatialSubModel):
                    if self.coregionalization_type != "spatial":
                        raise ValueError(
                            f"Model {model} is not of the same type as the first model (SpatialModel)"
                        )
                    # Check that the size of the Spatial fields are the same
                    if model.submodels[0].ns != self.n_spatial_nodes:
                        raise ValueError(
                            f"Model {model} is not of the same size as the first model (SpatialModel)"
                        )
                else:
                    raise ValueError(
                        "Invalid model type. Must be 'spatial' or 'spatio-temporal'."
                    )
                if len(model.submodels) > 1:
                    if isinstance(model.submodels[1], RegressionSubModel):
                        if (
                            model.submodels[1].n_fixed_effects
                            != self.n_fixed_effects_per_model
                        ):
                            raise ValueError(
                                f"Model {model} has a different number of fixed effects than the first model"
                            )

        # Get Models() hyperparameters
        theta: ArrayLike = []
        theta_keys: ArrayLike = []
        self.hyperparameters_idx: ArrayLike = [0]
        self.prior_hyperparameters: list[PriorHyperparameters] = []

        for model in self.models:
            theta_model = model.theta
            theta_keys_model = model.theta_keys

            # remove the theta that correspond to the "sigma_xx" where x can be whatever
            sigma_indices = [
                i
                for i, key in enumerate(theta_keys_model)
                if re.match(r"sigma_\w+", key)
            ]
            theta_model = [
                theta for i, theta in enumerate(theta_model) if i not in sigma_indices
            ]
            theta_keys_model = [
                key for i, key in enumerate(theta_keys_model) if i not in sigma_indices
            ]

            theta.append(xp.array(theta_model))
            theta_keys += theta_keys_model

            self.hyperparameters_idx.append(
                self.hyperparameters_idx[-1] + len(theta_model)
            )

            # Get the prior hyperparameters of the model
            self.prior_hyperparameters += [
                prior_hyperparameters
                for i, prior_hyperparameters in enumerate(model.prior_hyperparameters)
                if i not in sigma_indices
            ]

        # Initialize the Coregional Hyperparameters:
        (
            theta_coregional_model,
            theta_keys_coregional_model,
        ) = coregional_model_config.read_hyperparameters()
        theta.append(xp.array(theta_coregional_model))
        theta_keys += theta_keys_coregional_model

        self.hyperparameters_idx.append(
            self.hyperparameters_idx[-1] + len(theta_coregional_model)
        )

        # Finalize the hyperparameters
        self.theta: NDArray = xp.concatenate(theta)
        self.n_hyperparameters = self.theta.size
        self.theta_keys: NDArray = theta_keys

        print("theta: ", self.theta)
        print("theta_keys: ", self.theta_keys)

        # Initialize the Coregional Prior Hyperparameters
        for ph in coregional_model_config.ph_sigmas:
            if ph.type == "gaussian":
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        config=ph,
                    )
                )
            elif ph.type == "penalized_complexity":
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparameters(
                        config=ph,
                    )
                )
            else:
                raise ValueError(f"Invalid prior hyperparameters type: {ph.type}")

        for ph in coregional_model_config.ph_lambdas:
            if ph.type == "gaussian":
                self.prior_hyperparameters.append(
                    GaussianPriorHyperparameters(
                        config=ph,
                    )
                )
            elif ph.type == "penalized_complexity":
                self.prior_hyperparameters.append(
                    PenalizedComplexityPriorHyperparameters(
                        config=ph,
                    )
                )
            else:
                raise ValueError(f"Invalid prior hyperparameters type: {ph.type}")

        # Construct Coregional Model data from its Models()
        self.n_latent_parameters: int = 0
        self.latent_parameters_idx: list[int] = [0]
        self.n_observations: int = 0
        self.n_observations_idx: list[int] = [0]
        for model in self.models:
            self.n_latent_parameters += model.n_latent_parameters
            self.latent_parameters_idx.append(self.n_latent_parameters)
            self.n_observations += model.n_observations
            self.n_observations_idx.append(self.n_observations)

        # Assemble the latent parameters and observations from the Models()
        self.x: NDArray = xp.zeros(self.n_latent_parameters)
        self.y: NDArray = xp.zeros(self.n_observations)
        for i, model in enumerate(self.models):
            self.x[
                self.latent_parameters_idx[i] : self.latent_parameters_idx[i + 1]
            ] = model.x

            self.y[
                self.n_observations_idx[i] : self.n_observations_idx[i + 1]
            ] = model.y

        self.a: spmatrix = bdiag_tiling([model.a for model in self.models]).tocsc()
    
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
    
        print("memory used when model.a is assigned.")
        print("mem pool used: ", format_size(mempool.used_bytes()))           
        print("mem total bytes: ", format_size(mempool.total_bytes()))      
        
        # free memory -> delete model.a for all models
        for model in self.models:
            model.a = None
            model.y = None
            model.x = None
            
        print("memory used after model.a is deleted.")
        mempool.free_all_blocks()
        print("mem pool used: ", format_size(mempool.used_bytes()))           
        print("mem total bytes: ", format_size(mempool.total_bytes()))      
        

        if self.coregionalization_type == "spatio_temporal":
            self.permutation_Qst = self._generate_permutation_indices(
                self.n_temporal_nodes, self.n_spatial_nodes, self.n_models
            )

            self.permutation_latent_variables = (
                self._generate_permutation_indices_for_a(
                    self.n_temporal_nodes,
                    self.n_spatial_nodes,
                    self.n_models,
                    self.n_fixed_effects_per_model,
                )
            )

            self.a = self.a[:, self.permutation_latent_variables]
            self.x = self.x[self.permutation_latent_variables]

        elif self.coregionalization_type == "spatial":
            # permute fixed effects to the end
            self.permutation_latent_variables = (
                self._generate_permutation_indices_spatial(
                    self.n_spatial_nodes, self.n_fixed_effects_per_model, self.n_models
                )
            )

            self.a = self.a[:, self.permutation_latent_variables]
            self.x = self.x[self.permutation_latent_variables]

        # self.inverse_permutation_latent_variables = xp.argsort(self.permutation_latent_variables)
        # self.perm2 = self._generate_permutation_indices_for_a_new(
        #     self.n_temporal_nodes,
        #     self.n_spatial_nodes,
        #     self.n_models,
        #     self.n_fixed_effects_per_model,
        # )
        # self.inverse_permutation_latent_variables = xp.argsort(self.perm2)

        # --- Recurrent variables
        self.Q_prior_data_mapping = [0]
        
        self.rows_Qprior_re = None
        self.columns_Qprior_re = None
        self.data_Qprior_re = None

        self.permutation_vector_Q_prior = None
        self.permutation_indices_Q_prior = None
        self.permutation_indptr_Q_prior = None

        self.Q_conditional = None
        self.Q_conditional_data_mapping = [0]
        
        self.Q_prior: spmatrix = None # need this otherwise the construct will fail
        print("Calling construct_Q_prior in the init...")
        self.construct_Q_prior()


    def construct_Q_prior(self) -> spmatrix:
        # number of random effects per model
        n_re = self.n_spatial_nodes * self.n_temporal_nodes
        
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        # Qu_list: list = []
        # Q_r: list = []

        Qu_list: list = [None] * self.n_models          
        Q_r: list = [None] * self.n_models

        for i, model in enumerate(self.models):
            submodel_st = model.submodels[0]
            # Get the spatio-temporal submodel idx
            kwargs_st = {}
            for hp_idx in range(
                self.hyperparameters_idx[i], self.hyperparameters_idx[i + 1]
            ):
                kwargs_st[self.theta_keys[hp_idx]] = float(self.theta[hp_idx])

            # print("in construct Qprior: kwargs_st: ", kwargs_st)
            #Qu_list.append(submodel_st.construct_Q_prior(**kwargs_st).tocsc())
            Qu_list[i] = submodel_st.construct_Q_prior(**kwargs_st).tocsc()

            if len(model.submodels) > 1:
                # Create the regression tip
                submodel_r = model.submodels[1]
                # Get the spatio-temporal submodel idx
                kwargs_r = {}
                #Q_r.append(submodel_r.construct_Q_prior(**kwargs_r).tocsc())
                Q_r[i] = submodel_r.construct_Q_prior(**kwargs_r).tocsc()

        sigma_0 = xp.exp(self.theta[self.theta_keys.index("sigma_0")])
        sigma_1 = xp.exp(self.theta[self.theta_keys.index("sigma_1")])
        # print("sigma_0: ", sigma_0, "sigma_1: ", sigma_1)

        lambda_0_1 = self.theta[self.theta_keys.index("lambda_0_1")]

        if self.n_models == 2:
            q11 = sp.sparse.coo_matrix(
                (1 / sigma_0**2) * Qu_list[0]
                + (lambda_0_1**2 / sigma_1**2) * Qu_list[1]
            )
            if not q11.has_canonical_format:
                q11.sum_duplicates()  
            
            Qu_list[0] = None

            q21 = sp.sparse.coo_matrix((-lambda_0_1 / sigma_1**2) * Qu_list[1])
            if not q21.has_canonical_format:
                q21.sum_duplicates()  
                
            q12 = sp.sparse.coo_matrix((-lambda_0_1 / sigma_1**2) * Qu_list[1])
            if not q12.has_canonical_format:
                q12.sum_duplicates()  
                
            q22 = sp.sparse.coo_matrix((1 / sigma_1**2) * Qu_list[1])
            if not q22.has_canonical_format:
                q22.sum_duplicates()  
                
            Qu_list[1] = None

            if self.data_Qprior_re is None:
                q11_rows = q11.row
                q11_columns = q11.col

                q21_rows = q21.row + n_re
                q21_columns = q21.col

                q12_rows = q12.row
                q12_columns = q12.col + n_re

                q22_rows = q22.row + n_re
                q22_columns = q22.col + n_re
                
                self.rows_Qprior_re = xp.concatenate(
                    [q11_rows, q12_rows, q21_rows, q22_rows]
                )
                self.columns_Qprior_re = xp.concatenate(
                    [q11_columns, q12_columns, q21_columns, q22_columns]
                )
                
            self.data_Qprior_re = xp.concatenate(
                [q11.data, q12.data, q21.data, q22.data]
            )

            # Qprior_st = sp.sparse.bmat([[q11, q12], [q21, q22]]).tocsc()

        elif self.n_models == 3:
            sigma_2 = xp.exp(self.theta[self.theta_keys.index("sigma_2")])

            lambda_0_2 = self.theta[self.theta_keys.index("lambda_0_2")]
            lambda_1_2 = self.theta[self.theta_keys.index("lambda_1_2")]
            
            q11 = sp.sparse.coo_matrix(
                (1 / sigma_0**2) * Qu_list[0]
                + (lambda_0_1**2 / sigma_1**2) * Qu_list[1]
                + (lambda_1_2**2 / sigma_2**2) * Qu_list[2]
            )
            Qu_list[0] = None
            if not q11.has_canonical_format:
                q11.sum_duplicates()  

            q21 = sp.sparse.coo_matrix(
                (-lambda_0_1 / sigma_1**2) * Qu_list[1]
                + (lambda_0_2 * lambda_1_2 / sigma_2**2) * Qu_list[2]
            )
            if not q21.has_canonical_format:
                q21.sum_duplicates()  
                
            q31 = sp.sparse.coo_matrix(-lambda_1_2 / sigma_2**2 * Qu_list[2])
            if not q31.has_canonical_format:
                q31.sum_duplicates()  
                            
            q22 = sp.sparse.coo_matrix(
                (1 / sigma_1**2) * Qu_list[1]
                + (lambda_0_2**2 / sigma_2**2) * Qu_list[2]
            )
            if not q22.has_canonical_format:
                q22.sum_duplicates()  
            Qu_list[1] = None

            q32 = sp.sparse.coo_matrix(-lambda_0_2 / sigma_2**2 * Qu_list[2])
            if not q32.has_canonical_format:
                q32.sum_duplicates()     
            
            q33 = sp.sparse.coo_matrix((1 / sigma_2**2) * Qu_list[2])
            if not q33.has_canonical_format:
                q33.sum_duplicates()   
            Qu_list[2] = None
            
            # not the most elegant way but im afraid that without the copy it might break in some cases
            q12 = (q21.copy()).T
            if not q12.has_canonical_format:
                q12.sum_duplicates()    
                
            q13 = (q31.copy()).T
            if not q13.has_canonical_format:
                q13.sum_duplicates() 
               
            q23 = (q32.copy()).T
            if not q23.has_canonical_format:
                q23.sum_duplicates() 
                            
            mempool.free_all_blocks()
            print("mem pool used: ", format_size(mempool.used_bytes()))           
            print("mem total bytes: ", format_size(mempool.total_bytes()))             

            # we only need these indices once in the beginning
            # then they can be none again and we can only collect data array
            if self.data_Qprior_re is None:
                print("\nBefore concatenate calls... first time...")
                
                q11_rows = q11.row
                q11_columns = q11.col
                
                q21_rows = q21.row + n_re
                q21_columns = q21.col
                
                q31_rows = q31.row + 2 * n_re
                q31_columns = q31.col    
                
                q22_rows = q22.row + n_re
                q22_columns = q22.col + n_re
                
                q32_rows = q32.row + 2 * n_re
                q32_columns = q32.col + n_re
                
                q33_rows = q33.row + 2 * n_re
                q33_columns = q33.col + 2 * n_re 
                
                ## CAREFUL IF THIS IS NOT A "TRUE" COPY ...         
                q12_rows = q12.row
                q12_columns = q12.col + n_re
                
                q13_rows = q13.row
                q13_columns = q13.col + 2 * n_re
                
                q23_rows = q23.row + n_re
                q23_columns = q23.col + 2 * n_re
                    
                
                self.rows_Qprior_re = xp.concatenate(
                    [
                        q11_rows,
                        q12_rows,
                        q13_rows,
                        q21_rows,
                        q22_rows,
                        q23_rows,
                        q31_rows,
                        q32_rows,
                        q33_rows,
                    ]
                )
                self.columns_Qprior_re = xp.concatenate(
                    [
                        q11_columns,
                        q12_columns,
                        q13_columns,
                        q21_columns,
                        q22_columns,
                        q23_columns,
                        q31_columns,
                        q32_columns,
                        q33_columns,
                    ]
                )
            
            # this changes every time -> need to keep
            self.data_Qprior_re = xp.concatenate(
                [
                    q11.data,
                    q12.data,
                    q13.data,
                    q21.data,
                    q22.data,
                    q23.data,
                    q31.data,
                    q32.data,
                    q33.data,
                ]
            ) 
            
            mempool.free_all_blocks()

            # Qprior_st = sp.sparse.bmat(
            #     [[q11, q12, q13], [q21, q22, q23], [q31, q32, q33]]
            # ).tocsc()
            
            # Qprior_re = sp.sparse.coo_matrix((self.data_Qprior_re, (self.rows_Qprior_re, self.columns_Qprior_re)), shape=( self.n_models * n_re,  self.n_models * n_re))

        # Apply the permutation to the Qprior_st
        if self.coregionalization_type == "spatio_temporal":
            # Permute matrix
            # Qprior_st_perm = Qprior_st[self.permutation_Qst, :][:, self.permutation_Qst]

            if self.permutation_indices_Q_prior is None:
                self.Qprior_re_perm = sp.sparse.csc_matrix(
                    (self.n_models * n_re, self.n_models * n_re),
                    dtype=self.data_Qprior_re.dtype,
                )
                #perm = np.arange(Qprior_st.shape[0])
                self.set_data_array_permutation_indices(
                    self.permutation_Qst,
                    self.rows_Qprior_re,
                    self.columns_Qprior_re,
                    self.n_models * n_re,
                )
                
                # we only need to set these once
                self.Qprior_re_perm.indices = self.permutation_indices_Q_prior
                self.Qprior_re_perm.indptr = self.permutation_indptr_Q_prior
                
                # dont need these anymore
                self.rows_Qprior_re = None
                self.columns_Qprior_re = None
                
            print("\nAfter computing Qprior permutation indices...")     
            mempool.free_all_blocks()
            print("mem pool used: ", format_size(mempool.used_bytes()))           
            print("mem total bytes: ", format_size(mempool.total_bytes()))         

            self.Qprior_re_perm.data = self.data_Qprior_re[
                self.permutation_vector_Q_prior
            ]
                        
        else:
            # Qprior_st_perm = Qprior_st
            self.Qprior_re_perm = sp.sparse.coo_matrix(
                (self.data_Qprior_re, (self.rows_Qprior_re, self.columns_Qprior_re)),
                shape=(self.n_models * n_re, self.n_models * n_re),
            ).tocsc()

        if Q_r != []:
            if self.Q_prior is None:
                Qprior_reg = bdiag_tiling(Q_r).tocsc()
                self.Q_prior = bdiag_tiling([self.Qprior_re_perm, Qprior_reg]).tocsc()

                #self.Q_prior = bdiag_tiling([Qprior_st_perm, Qprior_reg]).tocsc()
            else:                
                mempool.free_all_blocks()
                   
                self.Q_prior.tocsc()
                self.Q_prior.sort_indices()
                self.Q_prior.data[: self.Qprior_re_perm.nnz] = self.Qprior_re_perm.data
                
                # Qprior_reg = bdiag_tiling(Q_r).tocsc()
                # self.Q_prior = bdiag_tiling([Qprior_st_perm, Qprior_reg]).tocsc()

        else:
            # self.Q_prior = self.Qprior_re_perm
            self.Q_prior = Qprior_st_perm
                         
        mempool.free_all_blocks()

        return self.Q_prior
    
    @time_range()
    def spgemm(self, A, B, rows: int = 5408):
        
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        # print("In custom spgemm. mem pool used: ", format_size(mempool.used_bytes()))             
        # print("In custom spgemm. mem total bytes: ", format_size(mempool.total_bytes()))  
        
        C = None
        for i in range(0, A.shape[0], rows):
            A_block = A[i:min(A.shape[0], i+rows)]
            C_block = A_block @ B
            if C is None:
                C = C_block
            else:
                C = cp.sparse.vstack([C, C_block], format="csr")
                
        # print("after custom sgemm call. mem pool used: ", format_size(mempool.used_bytes()))            
        # print("after custom sgemm call. mem total bytes: ", format_size(mempool.total_bytes()))         
        mempool.free_all_blocks()
        return C
    
    @time_range()
    def custom_Q_ATDA(self, Q: sp.sparse.csc_matrix, A: sp.sparse.csc_matrix, D_diag: xp.ndarray) -> sp.sparse.csr_matrix:
        """
        Computes A^T * D * A with minimal memory and maximum speed.
        - Uses sparse diagonal multiplication.
        - No temporary dense matrices.
        """
        
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

        DA = A.multiply(D_diag[:, xp.newaxis]).T.tocsr() 
        mempool.free_all_blocks()
        
        # use batched spgemm if mempool is full
        if mempool.used_bytes() > 80 * 1024**3:
            batch_size = int(xp.ceil(A.shape[0] / 2))
        else:
            batch_size = A.shape[0]

        #print("Calling custom spgemm... with rows: ", batch_size)
        ATDA = self.spgemm(DA, A, rows=batch_size)
        mempool.free_all_blocks()
        
        self.Qconditional = Q - ATDA  
        mempool.free_all_blocks()
    
        return self.Qconditional

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
        mempool = cp.get_default_memory_pool()
        
        #d_list = [None] * self.n_models
        d_vec = xp.zeros(self.n_observations)
        
        for i, model in enumerate(self.models):
            if model.likelihood_config.type == "gaussian":
                kwargs = {
                    "eta": eta[
                        self.n_observations_idx[i] : self.n_observations_idx[i + 1]
                    ],
                    "theta": float(self.theta[self.hyperparameters_idx[i + 1] - 1]),
                }
            else:
                kwargs = {
                    "eta": eta[
                        self.n_observations_idx[i] : self.n_observations_idx[i + 1]
                    ],
                }

            #d_list[i] = model.likelihood.evaluate_hessian_likelihood(**kwargs)
            d_vec[
                self.n_observations_idx[i] : self.n_observations_idx[i + 1]
            ] = model.likelihood.evaluate_hessian_likelihood(
                **kwargs
            ).diagonal()

        self.Qconditional = self.custom_Q_ATDA(
            Q=self.Q_prior,
            A=self.a,
            D_diag=d_vec,
        )
        self.Q_conditional = self.Qconditional.tocsc()
        mempool.free_all_blocks()       
        
        return self.Q_conditional

    def construct_information_vector(
        self,
        eta: NDArray,
        x_i: NDArray,
    ) -> NDArray:
        """Construct the information vector."""

        gradient_vector_list = []
        for i, model in enumerate(self.models):
            gradient_likelihood = model.likelihood.evaluate_gradient_likelihood(
                eta=eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                y=self.y[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                theta=float(self.theta[self.hyperparameters_idx[i + 1] - 1]),
            )

            gradient_vector_list.append(gradient_likelihood)

        gradient_likelihood = xp.concatenate(gradient_vector_list)

        information_vector: NDArray = (
            -1 * self.Q_prior @ x_i + self.a.T @ gradient_likelihood
        )

        return information_vector

    def is_likelihood_gaussian(self) -> bool:
        """Check if the likelihood is Gaussian."""
        for model in self.models:
            if not model.is_likelihood_gaussian():
                return False
        return True

    def evaluate_likelihood(
        self,
        eta: NDArray,
    ) -> float:
        likelihood: float = 0.0
        for i, model in enumerate(self.models):
            likelihood += model.likelihood.evaluate_likelihood(
                eta=eta[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                y=self.y[self.n_observations_idx[i] : self.n_observations_idx[i + 1]],
                theta=float(self.theta[self.hyperparameters_idx[i + 1] - 1]),
            )
            # print("theta: ", float(self.theta[self.hyperparameters_idx[i+1]-1]))

        return likelihood

    def evaluate_log_prior_hyperparameters(self) -> float:
        """Evaluate the log prior hyperparameters."""
        log_prior = 0.0

        theta_interpret = self.theta

        # print("\nin evaluate_log_prior_hyperparameters")
        for i, prior_hyperparameter in enumerate(self.prior_hyperparameters):
            # print(f"theta_interpret{i}: {theta_interpret[i]}, log prior: ", prior_hyperparameter.evaluate_log_prior(theta_interpret[i]))
            log_prior += prior_hyperparameter.evaluate_log_prior(theta_interpret[i])

        return log_prior

    def __str__(self) -> str:
        """String representation of the model."""
        # Collect general information about the model
        coregional_model_info = [
            " --- Coregional Model ---",
            f"n_hyperparameters: {self.n_hyperparameters}",
            f"n_latent_parameters: {self.n_latent_parameters}",
            f"n_observations: {self.n_observations}",
        ]

        # Collect each submodel's information
        model_info = [str(model) for model in self.models]

        # Combine model information and submodel information
        return "\n".join(coregional_model_info + model_info)

    def _generate_permutation_indices_spatial(
        self, n_spatial_nodes: int, n_fixed_effects_per_model: int, n_models: int
    ):
        perm_vec = np.zeros(
            n_models * (n_spatial_nodes + n_fixed_effects_per_model), dtype=int
        )
        for i in range(n_models):
            perm_vec[i * n_spatial_nodes : (i + 1) * n_spatial_nodes] = range(
                i * (n_spatial_nodes + n_fixed_effects_per_model),
                i * (n_spatial_nodes + n_fixed_effects_per_model) + n_spatial_nodes,
            )
            perm_vec[
                n_models * n_spatial_nodes
                + i * n_fixed_effects_per_model : n_models * n_spatial_nodes
                + (i + 1) * n_fixed_effects_per_model
            ] = range(
                i * (n_spatial_nodes + n_fixed_effects_per_model) + n_spatial_nodes,
                (i + 1) * (n_spatial_nodes + n_fixed_effects_per_model),
            )

        return perm_vec

    def _generate_permutation_indices(
        self, n_temporal_nodes: int, n_spatial_nodes: int, n_models: int
    ):
        """
        Generate a permutation vector containing indices in the pattern:
        [0:block_size, n*block_size:(n+1)*block_size, 1*block_size:(1+1)*block_size, (n+1)*block_size:(n+1+1)*block_size, ...]

        Parameters
        ----------
        n_temporal_nodes : int
            Number of blocks.
        n_spatial_nodes : int
            Size of each block.
        n_models : int
            Number of models.

        Returns
        -------
        np.ndarray
            The generated permutation vector.
        """
        indices = np.arange(n_temporal_nodes * n_spatial_nodes)

        first_idx = indices.reshape(n_temporal_nodes, n_spatial_nodes)
        second_idx = first_idx + n_temporal_nodes * n_spatial_nodes

        if n_models == 2:
            perm_vectorized = np.hstack((first_idx, second_idx)).flatten()
        if n_models == 3:
            third_idx = second_idx + n_temporal_nodes * n_spatial_nodes
            perm_vectorized = np.hstack((first_idx, second_idx, third_idx)).flatten()

        return perm_vectorized

    def _generate_permutation_indices_for_a_new(
        self,
        n_temporal_nodes: int,
        n_spatial_nodes: int,
        n_models: int,
        n_fixed_effects_per_model: int,
    ):
        """
        Generate a permutation vector containing indices in the pattern:
        [0:block_size, n*block_size:(n+1)*block_size, 1*block_size:(1+1)*block_size, (n+1)*block_size:(n+1+1)*block_size, ...]

        Parameters
        ----------
        n_temporal_nodes : int
            Number of blocks.
        n_spatial_nodes : int
            Size of each block.
        n_models : int
            Number of models.

        Returns
        -------
        np.ndarray
            The generated permutation vector.
        """
        indices = np.arange(n_temporal_nodes * n_spatial_nodes)
        # indices_fixed_effects_1 = len(indices) + np.arange(n_fixed_effects_per_model)

        first_idx = indices.reshape(n_temporal_nodes, n_spatial_nodes)
        second_idx = first_idx + n_temporal_nodes * n_spatial_nodes
        indices_fixed_effects_1 = 2 * n_temporal_nodes * n_spatial_nodes + np.arange(
            n_fixed_effects_per_model
        )
        indices_fixed_effects_2 = n_fixed_effects_per_model + indices_fixed_effects_1

        if n_models == 2:
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                ]
            )
        elif n_models == 3:
            third_idx = (
                second_idx
                + n_temporal_nodes * n_spatial_nodes
                + 2 * n_fixed_effects_per_model
            )
            indices_fixed_effects_3 = (
                3 * n_temporal_nodes * n_spatial_nodes
                + 2 * n_fixed_effects_per_model
                + np.arange(n_fixed_effects_per_model)
            )
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx, third_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                    indices_fixed_effects_3,
                ]
            )

        return perm_vectorized

    def _generate_permutation_indices_for_a(
        self,
        n_temporal_nodes: int,
        n_spatial_nodes: int,
        n_models: int,
        n_fixed_effects_per_model: int,
    ):
        """
        Generate a permutation vector containing indices in the pattern:
        [0:block_size, n*block_size:(n+1)*block_size, 1*block_size:(1+1)*block_size, (n+1)*block_size:(n+1+1)*block_size, ...]

        Parameters
        ----------
        n_temporal_nodes : int
            Number of blocks.
        n_spatial_nodes : int
            Size of each block.
        n_models : int
            Number of models.

        Returns
        -------
        np.ndarray
            The generated permutation vector.
        """
        indices = np.arange(n_temporal_nodes * n_spatial_nodes)
        indices_fixed_effects_1 = len(indices) + np.arange(n_fixed_effects_per_model)

        first_idx = indices.reshape(n_temporal_nodes, n_spatial_nodes)
        second_idx = (
            first_idx + n_fixed_effects_per_model + n_temporal_nodes * n_spatial_nodes
        )
        indices_fixed_effects_2 = (
            2 * n_temporal_nodes * n_spatial_nodes
            + n_fixed_effects_per_model
            + np.arange(n_fixed_effects_per_model)
        )

        if n_models == 2:
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                ]
            )
        elif n_models == 3:
            third_idx = (
                second_idx
                + n_temporal_nodes * n_spatial_nodes
                + n_fixed_effects_per_model
            )
            indices_fixed_effects_3 = (
                3 * n_temporal_nodes * n_spatial_nodes
                + 2 * n_fixed_effects_per_model
                + np.arange(n_fixed_effects_per_model)
            )
            perm_vectorized = np.concatenate(
                [
                    np.hstack((first_idx, second_idx, third_idx)).flatten(),
                    indices_fixed_effects_1,
                    indices_fixed_effects_2,
                    indices_fixed_effects_3,
                ]
            )

        return perm_vectorized

    def set_data_array_permutation_indices(
        self, permutation, a_rows: NDArray, a_cols: NDArray, n: int
    ) -> None:
        
        a_data_placeholder = xp.arange(0, len(a_rows), 1)
        a = sp.sparse.csc_matrix(
            sp.sparse.coo_matrix((a_data_placeholder, (a_rows, a_cols)), shape=(n, n), dtype=xp.float64)
        )
        
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        
        a_perm = a[permutation, :][:, permutation]
        
        mempool = cp.get_default_memory_pool()
        # print("mem pool used: ", format_size(mempool.used_bytes()))           
        # print("mem total bytes: ", format_size(mempool.total_bytes())) 
        
        # print("after cleaning up mempool")
        mempool.free_all_blocks()
        # print("mem pool used: ", format_size(mempool.used_bytes()))
        # print("mem total bytes: ", format_size(mempool.total_bytes()))
        
        self.permutation_vector_Q_prior = a_perm.data.astype(xp.int32)
        self.permutation_indices_Q_prior = a_perm.indices
        self.permutation_indptr_Q_prior = a_perm.indptr
                

    def get_solver_parameters(self) -> dict:
        """Get the solver parameters."""
        diagonal_blocksize = self.n_models * self.n_spatial_nodes
        n_diag_blocks = self.n_temporal_nodes
        arrowhead_blocksize = self.n_fixed_effects_per_model * self.n_models

        param = {
            "diagonal_blocksize": diagonal_blocksize,
            "n_diag_blocks": n_diag_blocks,
            "arrowhead_blocksize": arrowhead_blocksize,
        }

        return param
