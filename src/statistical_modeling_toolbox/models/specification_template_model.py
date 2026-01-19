"""
[Model Name] - Statistical Model Specification
===============================================

Component Type: Model (Latent Gaussian Field Structure)

Statistical Definition
----------------------
[Brief description of what this model represents]

[Typical use cases: temporal correlation, spatial structure, smoothness, etc.]


Mathematical form:

.. math::
    % For GMRF: x | \theta \sim N(\mu(x), Q(\theta)^{-1})
    % where Q(\theta) is the precision matrix
    % Example for AR1: x_t | x_{t-1}, \theta \sim N(\rho x_{t-1}, \tau^{-1})
    

Model structure:
    Dimension: [n = length of latent field]
    Mean structure: [Usually 0 for random effects, μ(X) for regression]
    Precision structure: Q(θ) [sparse, banded, block-structured, etc.]

Parameters:
    - [Latent field parameters, e.g., x_1, x_2, ..., x_n]

Hyperparameters:
    - tau (τ): Precision parameter (inverse variance)
    - [other model-specific hyperparameters, e.g., rho for AR, kappa for SPDE]


Properties
----------
Markov properties:
    [Conditional independence structure]
    [Neighborhood structure]

Sparsity pattern:
    Precision matrix Q: [describe pattern, e.g., tridiagonal, banded, k-neighbors]
    Number of non-zeros: O()
    Bandwidth: 

Constraints:
    [Sum-to-zero, identifiability constraints, etc.]
    

Statistical properties:
    Marginal variance: [if known analytically]
    Correlation structure: [describe decay, range, etc.]
    Stationarity: [stationary / non-stationary]
    Isotropy: [isotropic / anisotropic] (for spatial models)


Use Cases
---------
When to use this model:
    - 
    - 

Typical applications:
    - Temporal: [time series, survival analysis, etc.]
    - Spatial: [geostatistics, disease mapping, etc.]
    - Smoothing: [spline smoothing, trend estimation, etc.]

When NOT to use:
    - 
    - 

Alternatives:
    [Other models to consider and when to prefer them]


Parametrization
---------------
Standard Parametrization:
    [Conventional statistical parameters]
    σ² (variance), ρ (correlation), etc.

INLA/Precision Parametrization:
    τ = 1/σ² (precision)
    [transformed correlation parameters if needed]
    
Transformation formulas:
    .. math::
        \tau = 1 / \sigma^2
        % Add other transformations

Working space (for optimization):
    [Log-scale, logit-scale for bounded parameters]
    log(τ), log((1+ρ)/(1-ρ)), etc.


Required Methods
----------------
Core Methods (all models must implement):
    [x] precision_matrix(hyperparams) - Construct Q(θ)
    [x] log_density(x, params) - Evaluate log p(x | θ)
    [x] gradient(x, params, wrt='x') - Gradient for mode finding
    [x] hessian(x, params, wrt='x') - Hessian (usually equals -Q)
    [x] build(data, formula) - Construct model from data

Optional but recommended:
    [ ] constraint_matrix() - For identifiability (RW models, etc.)
    [ ] sample(n, hyperparams) - Generate samples from prior
    [ ] marginal_variance(hyperparams) - Analytical marginal variance

Method signatures:
    See implementation section below


Precision Matrix Structure
---------------------------
Matrix type:
    [ ] Dense (avoid if possible)
    [x] Sparse - Format: [CSR / CSC]
    [ ] Structured - Type: [Banded / Toeplitz / Kronecker / Block]

Sparsity pattern:
    Description: 
    Non-zeros per row: 
    Total non-zeros: O()
    
Structure details:
    [For banded: bandwidth and band structure]
    [For block: block size and block pattern]
    [For Kronecker: separable structure]

Construction approach:
    [ ] Direct construction (fill non-zeros)
    [ ] From difference operators
    [ ] Kronecker product
    [ ] Sum of sparse matrices

Special properties:
    [ ] Symmetric
    [ ] Positive definite
    [ ] M-matrix
    [ ] Banded
    [ ] Block-structured


Computational Considerations
-----------------------------
Computational complexity:
    build(): O()
    precision_matrix(): O()
    log_density(): O() [typically O(n) for sparse Q]
    gradient(): O()
    solve with Q: O() [depends on sparsity and solver]

Memory requirements:
    Storage for Q: O() [number of non-zeros]
    Workspace for factorization: O()

Solver recommendations:
    Direct solver: [Cholesky, LDL^T]
    Iterative solver: [CG, GMRES] (for very large problems)
    Preconditioner: [if needed]

Numerical stability:
    Conditioning: [well-conditioned / ill-conditioned for certain parameters]
    Regularization: [if needed, e.g., add ε to diagonal]


DALIA Integration
-----------------
Used by DALIA modules:
    - dalia.core.inla: Combines with likelihood for full model
    - dalia.mode_finding: Uses gradient/hessian for optimization
    - dalia.integration: Uses precision_matrix for Laplace approximation
    - dalia.post_model_fitting: Uses precision for marginal computation

What DALIA expects:
    - Precision matrix Q(θ) as scipy.sparse or StructuredMatrix
    - Gradient w.r.t. x for mode finding
    - Hessian at mode (often Hessian ≈ -Q for GMRF)
    - Constraint matrix A if sum-to-zero or other constraints needed

Combination with likelihood:
    Full model: y | x, θ ~ Likelihood(g^{-1}(Ax + offset), ϕ)
                x | θ ~ Model(0, Q(θ)^{-1})
    
Reparametrization:
    [How hyperparameters are reparametrized for optimization]
    [Jacobian adjustments if needed]


Validation
----------
Correctness checks:
    1. Precision matrix properties:
       - Symmetry: Q == Q.T
       - Positive definiteness: all eigenvalues > 0
       - Sparsity pattern correct
       
    2. Analytical tests:
       - Known special cases (e.g., white noise when ρ=0)
       - Comparison with theoretical covariance structure
       
    3. Gradient validation:
       - Finite difference check with tolerance rtol=1e-5
       - Hessian vs precision matrix consistency

Numerical tests:
    - Condition number of Q for various hyperparameters
    - Stability of Cholesky factorization
    - Accuracy of linear solves

Cross-validation:
    Compare with: [R-INLA, STAN, custom implementation, analytical solution]

Test cases:
    1. Simple case: [e.g., n=10, standard parameters]
    2. Edge case: [e.g., high correlation, large n]
    3. Constraint handling: [if applicable]


Example Usage
-------------
.. code-block:: python

    # Basic standalone usage
    from statistical_modeling_toolbox.models import ModelName
    
    # Create and build model
    model = ModelName(order=1)  # or other initialization parameters
    model.build(data={'locations': locations})  # or {'time': times}, etc.
    
    # Get precision matrix
    hyperparams = {'tau': 1.0, 'rho': 0.8}  # example hyperparameters
    Q = model.precision_matrix(hyperparams)
    print(f"Precision matrix: shape={Q.shape}, nnz={Q.nnz}")
    
    # Evaluate log-density (prior)
    x = np.random.randn(model.n)
    log_prior = model.log_density(x, params=hyperparams)
    
    # Get gradient (for mode finding)
    grad = model.gradient(x, params=hyperparams, wrt='x')
    
    # Sample from prior
    samples = model.sample(n=100, params=hyperparams)
    
    # Integration with DALIA
    from dalia.core import DALIA
    from statistical_modeling_toolbox.likelihoods import GaussianLikelihood
    
    likelihood = GaussianLikelihood()
    dalia = DALIA(
        model=model,
        likelihood=likelihood,
        data=data,
        formula='y ~ f(idx, model=ModelName(...))'
    )
    results = dalia.fit()
    
    # Access posterior
    print(f"Posterior mean: {results.latent_field.mean}")
    print(f"Posterior std: {results.latent_field.std}")
    print(f"Hyperparameters: {results.hyperparams}")


Implementation Notes
--------------------
Precision matrix construction:
    [Specific approach for this model]
    [Use scipy.sparse.diags, sparse.csr_matrix, etc.]
    [Exploit structure for efficiency]

Constraints:
    [If sum-to-zero: construct constraint matrix A]
    [How to handle in optimization]

Numerical considerations:
    [Parameter bounds to ensure positive definiteness]
    [Scaling recommendations]
    [Numerical issues to watch for]

Performance tips:
    [Caching strategy for Q if hyperparameters don't change]
    [Efficient updates if structure is similar]


Notes & Open Questions
----------------------
Implementation notes:
    - 

Mathematical properties to verify:
    - 

Numerical stability concerns:
    - 

Open questions:
    - 


References
----------
.. [1] Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and 
       Applications. Chapman & Hall/CRC. Chapter: [X]
.. [2] [Model-specific primary reference]
.. [3] [Additional references]


Related Models
--------------
See Also:
    [List related models in the toolbox]
    - SimilarModel: For comparison
    - ExtensionModel: Generalization of this model


Authors
-------
- [Author Name] <email@domain.com> ([Date])


Version History
---------------
- 0.1.0 ([Date]): Initial specification


Status
------
[ ] Specification complete
[ ] Mathematical properties verified
[ ] Precision matrix construction implemented
[ ] Gradient/Hessian implemented
[ ] Constraint handling implemented (if needed)
[ ] Gradient validation passed
[ ] Integration with DALIA tested
[ ] Performance optimized
[ ] Documentation complete
[ ] Ready for production

"""

# Standard library imports
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

# Third-party imports
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix, diags

# Local imports - Base class
from statistical_modeling_toolbox.models.base_model import BaseModel

# Local imports - Backend
# from backend.datastructures.matrix import StructuredMatrix
# from backend.linalg.solvers import LinearSolver

# Module exports
__all__ = [
    'ModelName',
    'ModelConfig',
]

# Module-level constants
DEFAULT_PRECISION = 1.0


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for [ModelName].
    
    Attributes
    ----------
    validate_inputs : bool
        Whether to validate inputs, default: True
    cache_precision : bool
        Whether to cache precision matrix, default: True
    numerical_tolerance : float
        Tolerance for numerical operations, default: 1e-10
    """
    validate_inputs: bool = True
    cache_precision: bool = True
    numerical_tolerance: float = 1e-10


# ============================================================================
# Main Implementation
# ============================================================================

class ModelName(BaseModel):
    """
    [Brief one-line description of the model]
    
    [Detailed description explaining:
    - What random process this models
    - When to use it (temporal/spatial/smoothing)
    - Key mathematical properties
    - How it integrates with DALIA]
    
    Mathematical Form
    -----------------
    .. math::
        % GMRF definition
        x | \\theta \\sim N(0, Q(\\theta)^{-1})
        
    where Q(θ) is [describe precision structure].
    
    For example, for an AR(1) model:
    .. math::
        x_t | x_{t-1}, \\theta \\sim N(\\rho x_{t-1}, \\tau^{-1})
    
    Parameters
    ----------
    param1 : Type
        Description (e.g., order for AR, degree for spline)
    param2 : Type, optional
        Description, default: value
    config : Optional[ModelConfig], default=None
        Configuration object
    
    Attributes
    ----------
    n : int
        Dimension of the latent field
    param1 : Type
        Model parameter
    _precision_matrix : Optional[sparse.spmatrix]
        Cached precision matrix
    _constraint_matrix : Optional[NDArray]
        Constraint matrix for identifiability
    _hyperparams_cache : Optional[Dict]
        Cached hyperparameters for precision matrix
    
    Examples
    --------
    >>> model = ModelName(order=1)
    >>> model.build(data={'time': time_points})
    >>> Q = model.precision_matrix(hyperparams={'tau': 1.0, 'rho': 0.8})
    >>> print(f"Precision matrix sparsity: {Q.nnz / Q.size * 100:.2f}%")
    
    Notes
    -----
    [Important notes about:]
    - Markov properties and conditional independence
    - Sparsity pattern and computational efficiency
    - Constraints and identifiability
    - Numerical stability considerations
    
    References
    ----------
    .. [1] Rue & Held (2005). GMRF, Chapter X.
    .. [2] Model-specific reference.
    
    See Also
    --------
    RelatedModel : Related model structure
    """
    
    def __init__(
        self,
        param1: Type,
        param2: Type = default_value,
        config: Optional[ModelConfig] = None
    ) -> None:
        """
        Initialize [ModelName].
        
        Parameters
        ----------
        param1 : Type
            Description
        param2 : Type, optional
            Description, default: value
        config : Optional[ModelConfig], default=None
            Configuration object
        
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        super().__init__()
        
        # Configuration
        if config is None:
            config = ModelConfig()
        self.config = config
        
        # Validate and store parameters
        if config.validate_inputs:
            self._validate_init_params(param1, param2)
        
        self.param1 = param1
        self.param2 = param2
        
        # Model structure (to be set by build())
        self.n = None  # Dimension of latent field
        self._is_built = False
        
        # Cached structures
        self._precision_matrix = None
        self._constraint_matrix = None
        self._hyperparams_cache = None
    
    def build(
        self,
        data: Dict[str, Any],
        formula: Optional[str] = None
    ) -> 'ModelName':
        """
        Build model structure from data.
        
        Constructs the model given data about locations, time points,
        covariates, etc. This determines the dimension n and any
        structural properties.
        
        Parameters
        ----------
        data : dict
            Dictionary containing data to build the model:
                For temporal models:
                    - 'time': NDArray, time points
                For spatial models:
                    - 'locations': NDArray, shape (n, d), spatial coordinates
                    - 'mesh': Optional mesh structure
                For regression models:
                    - 'covariates': NDArray, shape (n, p)
                    - 'design_matrix': Pre-computed design matrix
        formula : str, optional
            Formula string for model specification (R-style)
            Example: 'y ~ f(time, model=AR1())'
        
        Returns
        -------
        self : ModelName
            Returns self for method chaining
        
        Raises
        ------
        ValueError
            If required data is missing or invalid
        
        Examples
        --------
        >>> model = ModelName(order=1)
        >>> model.build(data={'time': np.arange(100)})
        >>> print(f"Model dimension: {model.n}")
        
        >>> # For spatial model
        >>> locations = np.random.randn(200, 2)
        >>> model.build(data={'locations': locations})
        """
        if self.config.validate_inputs:
            self._validate_build_data(data)
        
        # Extract relevant data and determine model dimension
        # Example for temporal model:
        # self.time = data['time']
        # self.n = len(self.time)
        
        # Example for spatial model:
        # self.locations = data['locations']
        # self.n = len(self.locations)
        
        # Build internal structures if needed
        # self._setup_neighborhood()
        # self._setup_constraints()
        
        self._is_built = True
        return self
    
    def precision_matrix(
        self,
        hyperparams: Dict[str, Any],
        sparse_format: str = 'csr'
    ) -> sparse.spmatrix:
        """
        Construct precision matrix Q(θ) for the GMRF.
        
        For GMRF: x | θ ~ N(0, Q(θ)^{-1})
        
        Parameters
        ----------
        hyperparams : dict
            Dictionary of hyperparameters:
                - 'tau': float, precision parameter (inverse variance)
                - [other model-specific hyperparameters]
                  e.g., 'rho' for AR models, 'kappa' for SPDE
        sparse_format : {'csr', 'csc', 'coo'}, default='csr'
            Sparse matrix format
        
        Returns
        -------
        Q : sparse.spmatrix
            Precision matrix, shape (n, n)
            Sparse matrix exploiting the Markov structure
        
        Raises
        ------
        RuntimeError
            If model has not been built yet (call build() first)
        ValueError
            If hyperparameters are invalid
        
        Notes
        -----
        This is the core method that defines the model structure.
        The precision matrix encodes the conditional independence
        structure through its sparsity pattern.
        
        The result is cached if config.cache_precision is True and
        hyperparameters haven't changed.
        
        Examples
        --------
        >>> Q = model.precision_matrix(hyperparams={'tau': 1.0, 'rho': 0.9})
        >>> print(f"Shape: {Q.shape}, Non-zeros: {Q.nnz}")
        >>> print(f"Sparsity: {Q.nnz / Q.size * 100:.2f}%")
        
        >>> # Check properties
        >>> assert np.allclose(Q.toarray(), Q.T.toarray())  # Symmetric
        >>> assert np.all(np.linalg.eigvalsh(Q.toarray()) > 0)  # Positive definite
        """
        if not self._is_built:
            raise RuntimeError("Model must be built before computing precision matrix. Call build() first.")
        
        if self.config.validate_inputs:
            self._validate_hyperparams(hyperparams)
        
        # Check cache
        if self.config.cache_precision and self._hyperparams_cache == hyperparams:
            return self._precision_matrix
        
        # Extract hyperparameters
        tau = hyperparams['tau']
        # other_param = hyperparams.get('other_param', default)
        
        # Construct precision matrix
        # Example for AR1 model:
        # Q = tau * self._construct_ar1_precision(rho)
        
        # Example for RW2 model:
        # D = self._construct_difference_operator(order=2)
        # Q = tau * (D.T @ D)
        
        # Example for SPDE model:
        # Q = self._construct_spde_precision(tau, kappa, mesh)
        
        # Placeholder - IMPLEMENT ACTUAL CONSTRUCTION
        Q = sparse.eye(self.n, format=sparse_format) * tau
        
        # Cache if enabled
        if self.config.cache_precision:
            self._precision_matrix = Q
            self._hyperparams_cache = hyperparams.copy()
        
        return Q
    
    def constraint_matrix(self) -> Optional[NDArray]:
        """
        Return linear constraint matrix for identifiability.
        
        For models with flat priors (e.g., random walk), we need
        sum-to-zero constraints: A @ x = 0
        
        Returns
        -------
        A : NDArray or None
            Constraint matrix, shape (k, n) where k is number of constraints
            None if no constraints needed
            
            Common constraints:
            - Sum-to-zero: A = ones(1, n) / sqrt(n)
            - Differences: For higher-order random walks
        
        Notes
        -----
        Used by DALIA in:
        - Mode finding: constrained optimization
        - Integration: proper posterior computation
        
        For most models this returns None. Implement only if your model
        requires explicit constraints for identifiability.
        
        Examples
        --------
        >>> A = model.constraint_matrix()
        >>> if A is not None:
        ...     # Constraint: A @ x = 0
        ...     print(f"Number of constraints: {A.shape[0]}")
        """
        # Most models don't need explicit constraints
        # Implement if needed for your specific model
        # Example for sum-to-zero:
        # return np.ones((1, self.n)) / np.sqrt(self.n)
        
        return None
    
    def log_density(
        self,
        x: NDArray,
        params: Dict[str, Any]
    ) -> float:
        """
        Evaluate log-density (log-prior) of the latent field.
        
        Computes: log p(x | θ) = -1/2 * (x^T Q x + log|Q| + n*log(2π))
        
        Parameters
        ----------
        x : NDArray, shape (n,)
            Latent field values
        params : dict
            Hyperparameters (same as precision_matrix)
        
        Returns
        -------
        log_p : float
            Log-density value
        
        Notes
        -----
        For GMRF with precision Q:
        log p(x | θ) = (1/2) log|Q| - (1/2) x^T Q x - (n/2) log(2π)
        
        The log-determinant term can be expensive. For efficiency:
        - Use sparse Cholesky: log|Q| = 2 * sum(log(diag(L)))
        - Or compute analytically if structure allows
        
        Examples
        --------
        >>> x = np.random.randn(model.n)
        >>> log_p = model.log_density(x, params={'tau': 1.0, 'rho': 0.8})
        """
        if not self._is_built:
            raise RuntimeError("Model must be built first.")
        
        if len(x) != self.n:
            raise ValueError(f"x must have length {self.n}, got {len(x)}")
        
        # Get precision matrix
        Q = self.precision_matrix(params)
        
        # Compute quadratic form: x^T Q x
        quad_form = x @ (Q @ x)
        
        # Compute log-determinant
        # For efficiency, use Cholesky if available
        # from scipy.sparse.linalg import splu
        # lu = splu(Q.tocsc())
        # log_det_Q = np.sum(np.log(np.abs(lu.U.diagonal())))
        
        # Placeholder - implement efficient log-det computation
        log_det_Q = 0.0  # IMPLEMENT
        
        # Log-density
        log_p = 0.5 * log_det_Q - 0.5 * quad_form - 0.5 * self.n * np.log(2 * np.pi)
        
        return log_p
    
    def gradient(
        self,
        x: NDArray,
        params: Dict[str, Any],
        wrt: str = 'x'
    ) -> Union[NDArray, Dict[str, float]]:
        """
        Compute gradient of log-density.
        
        Parameters
        ----------
        x : NDArray, shape (n,)
            Latent field values
        params : dict
            Hyperparameters
        wrt : {'x', 'hyperparams'}, default='x'
            Compute gradient with respect to:
            - 'x': gradient w.r.t. latent field (for mode finding)
            - 'hyperparams': gradient w.r.t. hyperparameters (for optimization)
        
        Returns
        -------
        gradient : NDArray or dict
            If wrt='x': NDArray, shape (n,), gradient vector
            If wrt='hyperparams': dict with gradient for each hyperparam
        
        Notes
        -----
        For GMRF, gradient w.r.t. x is simple:
        ∇_x log p(x | θ) = -Q @ x
        
        Gradient w.r.t. hyperparameters requires derivatives of Q and log|Q|.
        
        Examples
        --------
        >>> grad_x = model.gradient(x, params={'tau': 1.0}, wrt='x')
        >>> grad_theta = model.gradient(x, params={'tau': 1.0}, wrt='hyperparams')
        >>> print(f"Gradient w.r.t. tau: {grad_theta['tau']}")
        """
        if wrt not in ['x', 'hyperparams']:
            raise ValueError(f"wrt must be 'x' or 'hyperparams', got {wrt}")
        
        if wrt == 'x':
            # For GMRF: ∇_x log p(x | θ) = -Q @ x
            Q = self.precision_matrix(params)
            return -(Q @ x)
        
        else:  # wrt == 'hyperparams'
            # Compute gradient w.r.t. each hyperparameter
            # This requires ∂Q/∂θ and ∂log|Q|/∂θ
            
            # Placeholder - implement hyperparameter gradients
            grad_dict = {}
            for key in params:
                grad_dict[key] = 0.0  # IMPLEMENT
            
            return grad_dict
    
    def hessian(
        self,
        x: NDArray,
        params: Dict[str, Any],
        wrt: str = 'x'
    ) -> Union[sparse.spmatrix, NDArray]:
        """
        Compute Hessian matrix of log-density.
        
        Parameters
        ----------
        x : NDArray, shape (n,)
            Latent field values
        params : dict
            Hyperparameters
        wrt : {'x', 'hyperparams'}, default='x'
            Compute Hessian with respect to:
            - 'x': Hessian w.r.t. latent field
            - 'hyperparams': Hessian w.r.t. hyperparameters
        
        Returns
        -------
        hessian : sparse.spmatrix or NDArray
            If wrt='x': sparse matrix, shape (n, n), equals -Q
            If wrt='hyperparams': dense matrix for second derivatives
        
        Notes
        -----
        For GMRF, Hessian w.r.t. x is constant:
        ∇²_x log p(x | θ) = -Q(θ)
        
        This is the key simplification that makes INLA efficient.
        
        Examples
        --------
        >>> H = model.hessian(x, params={'tau': 1.0}, wrt='x')
        >>> # For GMRF, H should equal -Q
        >>> Q = model.precision_matrix(params={'tau': 1.0})
        >>> assert np.allclose(H.toarray(), -Q.toarray())
        """
        if wrt == 'x':
            # For GMRF: Hessian = -Q (constant, doesn't depend on x)
            Q = self.precision_matrix(params)
            return -Q
        
        else:  # wrt == 'hyperparams'
            # Second derivatives w.r.t. hyperparameters
            # Typically not needed for INLA but useful for optimization
            
            # Placeholder
            n_params = len(params)
            return np.zeros((n_params, n_params))  # IMPLEMENT
    
    def sample(
        self,
        n_samples: int,
        params: Dict[str, Any],
        random_state: Optional[int] = None
    ) -> NDArray:
        """
        Generate random samples from the model prior.
        
        Samples from: x | θ ~ N(0, Q(θ)^{-1})
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        params : dict
            Hyperparameters
        random_state : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        samples : NDArray, shape (n_samples, n)
            Random samples from the prior
        
        Notes
        -----
        For GMRF, sampling can be done by:
        1. Cholesky: Q = L L^T, then x = L^{-T} z where z ~ N(0, I)
        2. Or: Compute Σ = Q^{-1} (expensive for large n)
        
        For sparse Q, use sparse Cholesky factorization.
        
        Examples
        --------
        >>> samples = model.sample(n_samples=100, params={'tau': 1.0}, random_state=42)
        >>> print(f"Sample mean: {samples.mean(axis=0)}")
        >>> print(f"Sample covariance: {np.cov(samples.T)}")
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Get precision matrix
        Q = self.precision_matrix(params)
        
        # Sample using sparse Cholesky
        from scipy.sparse.linalg import spsolve, splu
        
        samples = np.zeros((n_samples, self.n))
        
        # Factorize Q = L L^T
        try:
            lu = splu(Q.tocsc())
            
            for i in range(n_samples):
                # Generate z ~ N(0, I)
                z = np.random.randn(self.n)
                
                # Solve L^T x = z for x
                # This gives x ~ N(0, Q^{-1})
                samples[i] = lu.solve(z, trans='T')
        
        except Exception as e:
            raise RuntimeError(f"Failed to sample from model: {e}")
        
        return samples
    
    def marginal_variance(self, params: Dict[str, Any]) -> Union[float, NDArray]:
        """
        Compute marginal variance (if available analytically).
        
        Parameters
        ----------
        params : dict
            Hyperparameters
        
        Returns
        -------
        variance : float or NDArray
            Marginal variance (scalar if stationary, vector if not)
        
        Notes
        -----
        For stationary models, marginal variance is constant.
        For non-stationary models (e.g., RW), it depends on position.
        
        This is optional but useful for interpretation and initialization.
        
        Examples
        --------
        >>> var = model.marginal_variance(params={'tau': 1.0, 'rho': 0.8})
        >>> print(f"Marginal variance: {var}")
        """
        # Implement if analytical form is available
        # Example for AR1: Var(x_t) = τ^{-1} / (1 - ρ²)
        raise NotImplementedError("marginal_variance not implemented for this model")
    
    # ========================================================================
    # Validation and utility methods
    # ========================================================================
    
    def _validate_init_params(self, param1: Any, param2: Any) -> None:
        """Validate initialization parameters."""
        # Implement parameter validation
        # Example: check order > 0, check ranges, etc.
        pass
    
    def _validate_build_data(self, data: Dict[str, Any]) -> None:
        """Validate data for build()."""
        # Check required keys present
        # Check data types and shapes
        pass
    
    def _validate_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Validate hyperparameters."""
        # Check required hyperparameters
        if 'tau' not in hyperparams:
            raise ValueError("Hyperparameters must include 'tau'")
        
        # Check parameter constraints
        if hyperparams['tau'] <= 0:
            raise ValueError("Precision tau must be positive")
        
        # Model-specific validation
        # Example: check |rho| < 1 for AR models
    
    def __repr__(self) -> str:
        """String representation."""
        if self._is_built:
            return f"{self.__class__.__name__}(param1={self.param1}, n={self.n})"
        else:
            return f"{self.__class__.__name__}(param1={self.param1}, not built)"


# ============================================================================
# Helper Functions
# ============================================================================

def _construct_difference_operator(n: int, order: int = 1) -> sparse.spmatrix:
    """
    Construct difference operator matrix.
    
    Parameters
    ----------
    n : int
        Dimension
    order : int, default=1
        Order of differences (1 for first-order, 2 for second-order)
    
    Returns
    -------
    D : sparse.spmatrix
        Difference operator, shape (n-order, n)
    
    Examples
    --------
    >>> D1 = _construct_difference_operator(10, order=1)  # First differences
    >>> D2 = _construct_difference_operator(10, order=2)  # Second differences
    """
    if order == 1:
        # First-order differences: D1[i] = x[i+1] - x[i]
        diag_data = np.array([-np.ones(n-1), np.ones(n-1)])
        diag_offsets = np.array([0, 1])
        D = diags(diag_data, diag_offsets, shape=(n-1, n), format='csr')
    
    elif order == 2:
        # Second-order differences: D2[i] = x[i+2] - 2*x[i+1] + x[i]
        diag_data = np.array([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)])
        diag_offsets = np.array([0, 1, 2])
        D = diags(diag_data, diag_offsets, shape=(n-2, n), format='csr')
    
    else:
        raise NotImplementedError(f"Difference operator of order {order} not implemented")
    
    return D


def _standard_to_precision_params(params_std: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert standard parametrization to precision parametrization.
    
    Parameters
    ----------
    params_std : dict
        Standard parameters, e.g., {'sigma': 2.0}
    
    Returns
    -------
    params_prec : dict
        Precision parameters, e.g., {'tau': 0.25}
    
    Examples
    --------
    >>> params_prec = _standard_to_precision_params({'sigma': 2.0})
    >>> # Returns {'tau': 0.25}  (tau = 1/sigma^2)
    """
    params_prec = {}
    
    if 'sigma' in params_std:
        params_prec['tau'] = 1.0 / (params_std['sigma'] ** 2)
    
    # Copy other parameters unchanged
    for key, value in params_std.items():
        if key != 'sigma' and key not in params_prec:
            params_prec[key] = value
    
    return params_prec


# ============================================================================
# Validation utilities (for testing)
# ============================================================================

def validate_precision_matrix_properties(Q: sparse.spmatrix, tol: float = 1e-10) -> Dict[str, bool]:
    """
    Validate mathematical properties of precision matrix.
    
    Parameters
    ----------
    Q : sparse.spmatrix
        Precision matrix to validate
    tol : float, default=1e-10
        Tolerance for checks
    
    Returns
    -------
    results : dict
        Dictionary with validation results:
        - 'symmetric': bool
        - 'positive_definite': bool
        - 'sparse': bool
    
    Examples
    --------
    >>> Q = model.precision_matrix(hyperparams={'tau': 1.0})
    >>> results = validate_precision_matrix_properties(Q)
    >>> assert all(results.values()), "Precision matrix validation failed"
    """
    results = {}
    
    # Check symmetry
    Q_dense = Q.toarray()
    results['symmetric'] = np.allclose(Q_dense, Q_dense.T, atol=tol)
    
    # Check positive definiteness (all eigenvalues > 0)
    try:
        eigvals = np.linalg.eigvalsh(Q_dense)
        results['positive_definite'] = np.all(eigvals > -tol)
    except:
        results['positive_definite'] = False
    
    # Check sparsity (should be much sparser than dense)
    results['sparse'] = Q.nnz < 0.5 * Q.size
    
    return results


# ============================================================================
# End of module
# ============================================================================
