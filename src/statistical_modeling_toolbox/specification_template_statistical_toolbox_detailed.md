# Component: [Name] (Model/Likelihood/Prior)
> **Comprehensive specification for statistical modeling components**  
> *Complete documentation for models, likelihoods, and priors in the statistical toolbox*

---

## 1. Statistical Definition

### 1.1 Component Type
*Select one:*
- [ ] **Model** - Defines structure for latent field (e.g., AR, RW, SPDE)
- [ ] **Likelihood** - Links observations to latent field (e.g., Gaussian, Poisson)
- [ ] **Prior** - Defines prior distribution on hyperparameters (e.g., log-gamma, PC prior)

### 1.2 Mathematical Form

#### Probability Density/Mass Function
*Complete mathematical definition:*
$$
% For likelihood: p(y | \eta, \theta)
% For model: \mathbf{x} | \theta \sim \text{Model}(\theta)
% For prior: \pi(\theta)
$$

**Notation table:**
| Symbol | Description | Type | Dimension |
|--------|-------------|------|-----------|
| | | scalar/vector/matrix | |

#### Parameters and Their Interpretation
**Parameters** (usually part of the latent field):
| Parameter | Symbol | Interpretation | Domain | Default |
|-----------|--------|----------------|--------|---------|
| | | | | |

**Hyperparameters** (control the distribution):
| Hyperparameter | Symbol | Interpretation | Domain | Default |
|----------------|--------|----------------|--------|---------|
| | | | | |

### 1.3 Statistical Properties

#### Support
*Domain where density is defined:*
- For continuous: 
- For discrete: 
- Boundary behavior: 

#### Moments
*First and second moments (if available):*
$$
\mathbb{E}[X] = 
$$
$$
\text{Var}[X] = 
$$

*Higher moments (if relevant):*
- Skewness: 
- Kurtosis: 

#### Special Properties
- [ ] **Conjugacy:** Conjugate to (if applicable)
- [ ] **Exponential family:** Yes / No
- [ ] **Scale family:** Yes / No
- [ ] **Location-scale family:** Yes / No
- [ ] **Markov property:** Yes / No (relevant for models)
- [ ] **Conditional independence structure:** (describe)

### 1.4 Literature References

**Primary references:**
1. Author. (Year). *Title*. Journal/Book. DOI/Link
   - Key contribution: 

**INLA-specific references:**
1. Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields*. Chapter: 
2. Other INLA papers: 

**Additional reading:**
- 

---

## 2. Use Cases and Applications

### 2.1 When to Use This Component
*Describe scenarios where this is appropriate:*


*Data characteristics that suggest this choice:*
- 

### 2.2 Typical Applications
*Common application domains:*
1. **Domain 1:** Description
2. **Domain 2:** Description

*Example problems:*
- 

### 2.3 Limitations and Alternatives

**When NOT to use:**
- Condition 1: 
- Condition 2: 

**Alternative components:**
| Alternative | When to prefer | Trade-offs |
|-------------|----------------|------------|
| | | |

---

## 3. Parametrization

### 3.1 Standard Parametrization
*Conventional statistical parametrization:*

**Parameters:**
$$
\theta_{\text{standard}} = (\theta_1, \theta_2, \ldots)
$$

*Description of each parameter:*
- $\theta_1$: 
- $\theta_2$: 

*Interpretation and scale:*


### 3.2 INLA/Precision Parametrization
*INLA often works with precision (inverse variance) parametrization:*

**Precision parametrization:**
$$
\theta_{\text{INLA}} = (\tau, \ldots)
$$

*Where:*
- $\tau = 1/\sigma^2$ (precision)
- 

**Transformation between parametrizations:**
$$
\theta_{\text{INLA}} = T(\theta_{\text{standard}})
$$

*Explicit transformation:*
```python
def standard_to_inla(params_std):
    """Convert standard parameters to INLA parametrization."""
    # Example: tau = 1 / sigma**2
    return params_inla
```

### 3.3 Link Functions (For Likelihoods Only)

**Canonical link function:**
$$
g(\mu) = \eta
$$

*Where:*
- $\mu$: Expected value
- $\eta$: Linear predictor
- $g$: Link function

**Inverse link function:**
$$
\mu = g^{-1}(\eta)
$$

**Common link functions for this likelihood:**
| Link Name | Function $g(\mu)$ | Inverse $g^{-1}(\eta)$ | When to use |
|-----------|-------------------|------------------------|-------------|
| | | | |

**Derivatives:**
$$
\frac{dg^{-1}(\eta)}{d\eta} = 
$$

### 3.4 Reparametrization for Optimization
*Does the component need reparametrization for numerical stability?*
- [ ] Yes (describe below)
- [ ] No

*If yes, describe the reparametrization:*


*Working space vs natural space:*
- Working space (for optimization): 
- Natural space (for interpretation): 

---

## 4. Required Methods and Interface

### 4.1 Base Class Inheritance
```python
# For Models
from statistical_modeling_toolbox.models.base_model import BaseModel

# For Likelihoods
from statistical_modeling_toolbox.likelihoods.likelihood import Likelihood

# For Priors
from statistical_modeling_toolbox.priors.prior import Prior
```

### 4.2 Core Methods (All Components)

#### Method 1: Log-Density Evaluation
```python
def log_density(
    self,
    x: Union[float, NDArray],
    params: Dict[str, Any]
) -> Union[float, NDArray]:
    """
    Evaluate log-density at x given parameters.
    
    Args:
        x: Evaluation point(s), shape (n,) or scalar
        params: Dictionary of parameters
            - param1: Description
            - param2: Description
    
    Returns:
        Log-density value(s), same shape as x
        
    Notes:
        - Returns log p(x | params)
        - Should handle both scalar and vector inputs
        - Must be numerically stable (avoid overflow/underflow)
    
    Examples:
        >>> component = ComponentName(...)
        >>> log_p = component.log_density(x=2.5, params={'mu': 0, 'tau': 1})
    """
    pass
```

#### Method 2: Gradient Computation
```python
def gradient(
    self,
    x: Union[float, NDArray],
    params: Dict[str, Any],
    wrt: str = 'x'
) -> Union[float, NDArray]:
    """
    Compute gradient of log-density.
    
    Args:
        x: Evaluation point(s)
        params: Dictionary of parameters
        wrt: Compute gradient with respect to:
            - 'x': gradient w.r.t. x (for mode finding)
            - 'params': gradient w.r.t. parameters (for optimization)
    
    Returns:
        Gradient vector or matrix
        
    Notes:
        - For wrt='x': returns ∇_x log p(x | params)
        - For wrt='params': returns ∇_θ log p(x | params)
        - Must be consistent with log_density
    
    Examples:
        >>> grad = component.gradient(x=2.5, params={'mu': 0, 'tau': 1}, wrt='x')
    """
    pass
```

#### Method 3: Hessian Computation
```python
def hessian(
    self,
    x: Union[float, NDArray],
    params: Dict[str, Any],
    wrt: str = 'x'
) -> NDArray:
    """
    Compute Hessian matrix of log-density.
    
    Args:
        x: Evaluation point(s)
        params: Dictionary of parameters
        wrt: Compute Hessian with respect to:
            - 'x': Hessian w.r.t. x
            - 'params': Hessian w.r.t. parameters
            - 'mixed': Mixed derivatives
    
    Returns:
        Hessian matrix, shape depends on wrt
        
    Notes:
        - For wrt='x': returns ∇²_x log p(x | params)
        - Often sparse or structured for models
        - May return structure information (e.g., band structure)
    
    Examples:
        >>> H = component.hessian(x=values, params={'mu': 0, 'tau': 1})
    """
    pass
```

### 4.3 Model-Specific Methods

#### Method 4: Precision Matrix (Models Only)
```python
def precision_matrix(
    self,
    hyperparams: Dict[str, Any],
    n: Optional[int] = None
) -> Union[NDArray, sparse.spmatrix, StructuredMatrix]:
    """
    Compute precision matrix Q for the model.
    
    For GMRF: x | θ ~ N(0, Q(θ)^{-1})
    
    Args:
        hyperparams: Dictionary of hyperparameters
            - tau: precision parameter
            - other: model-specific hyperparameters
        n: Size of the field (if not specified at init)
    
    Returns:
        Precision matrix Q, shape (n, n)
        Type depends on structure (dense, sparse, structured)
        
    Notes:
        - Q is typically sparse for GMRF models
        - Return scipy.sparse format or custom structured matrix
        - Include structure information if available
    
    Examples:
        >>> Q = model.precision_matrix(hyperparams={'tau': 1.0, 'alpha': 2})
    """
    pass
```

#### Method 5: Constraint Matrix (Models Only)
```python
def constraint_matrix(self) -> Optional[NDArray]:
    """
    Return linear constraint matrix for sum-to-zero constraints.
    
    For models requiring identifiability constraints: A @ x = 0
    
    Returns:
        Constraint matrix A, shape (k, n) where k is number of constraints
        None if no constraints needed
        
    Notes:
        - Common for random walk models
        - Used in mode finding and integration
    
    Examples:
        >>> A = model.constraint_matrix()
        >>> # A @ x = 0 enforces sum-to-zero
    """
    pass
```

#### Method 6: Build Method (Models Only)
```python
def build(
    self,
    data: Dict[str, Any],
    formula: Optional[str] = None
) -> 'Model':
    """
    Build model structure from data and formula.
    
    Args:
        data: Dictionary containing relevant data
            - 'locations': For spatial models
            - 'time': For temporal models
            - 'covariates': For regression models
        formula: Optional formula string (R-style or similar)
    
    Returns:
        Self (for method chaining)
        
    Notes:
        - Constructs model structure
        - Determines dimensions
        - Sets up constraints if needed
    
    Examples:
        >>> model = AR1Model().build(data={'time': time_points})
    """
    pass
```

### 4.4 Likelihood-Specific Methods

#### Method 7: Link Function (Likelihoods Only)
```python
def link_function(self, mu: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Apply link function: η = g(μ)
    
    Args:
        mu: Expected value (mean parameter)
    
    Returns:
        Linear predictor η
        
    Examples:
        >>> likelihood = PoissonLikelihood()
        >>> eta = likelihood.link_function(mu=5.0)  # log(5.0)
    """
    pass

def inverse_link_function(self, eta: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Apply inverse link function: μ = g^{-1}(η)
    
    Args:
        eta: Linear predictor
    
    Returns:
        Expected value μ
        
    Examples:
        >>> mu = likelihood.inverse_link_function(eta=1.6)
    """
    pass
```

#### Method 8: Deviance (Likelihoods Only)
```python
def deviance(
    self,
    observed: NDArray,
    predicted: NDArray
) -> float:
    """
    Compute deviance for model comparison.
    
    Args:
        observed: Observed data y
        predicted: Predicted values ŷ
    
    Returns:
        Deviance value
        
    Notes:
        - Used in DIC, WAIC calculations
        - Lower is better
    """
    pass
```

### 4.5 Optional Methods

#### Method 9: Sampling (Optional)
```python
def sample(
    self,
    n: int,
    params: Dict[str, Any],
    random_state: Optional[int] = None
) -> NDArray:
    """
    Generate random samples.
    
    Args:
        n: Number of samples
        params: Distribution parameters
        random_state: Random seed for reproducibility
    
    Returns:
        Samples, shape (n,) or (n, d)
        
    Notes:
        - Useful for simulation studies
        - For validation and testing
    """
    pass
```

#### Method 10: Moments (Optional)
```python
def moments(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute analytical moments if available.
    
    Args:
        params: Distribution parameters
    
    Returns:
        Dictionary with 'mean', 'variance', etc.
    """
    pass
```

### 4.6 Method Summary Table
| Method | Required | Models | Likelihoods | Priors | Purpose |
|--------|----------|--------|-------------|--------|---------|
| `log_density` | Yes | Yes | Yes | Yes | Evaluate log-density |
| `gradient` | Yes | Yes | Yes | Yes | First derivatives |
| `hessian` | Yes | Yes | Yes | Yes | Second derivatives |
| `precision_matrix` | Yes | Yes | No | No | GMRF precision |
| `constraint_matrix` | If needed | Yes | No | No | Identifiability |
| `build` | Yes | Yes | No | No | Construct from data |
| `link_function` | Yes | No | Yes | No | Link/inverse link |
| `deviance` | Yes | No | Yes | No | Model comparison |
| `sample` | Optional | Optional | Optional | Optional | Random generation |
| `moments` | Optional | Optional | Optional | Optional | Analytical moments |

---

## 5. Implementation Specifications

### 5.1 Class Structure
```python
from typing import Dict, Any, Optional, Union
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

class ComponentName(BaseClass):
    """
    [Component type]: [Brief description]
    
    [Detailed description of what this component does, when to use it,
    and any important mathematical or computational properties]
    
    Mathematical Form:
        [LaTeX equation in docstring]
    
    Parameters:
        [List parameters and their meanings]
    
    Attributes:
        attr1 (type): Description
        attr2 (type): Description
    
    Examples:
        >>> component = ComponentName(param1=value1)
        >>> log_p = component.log_density(x, params)
    
    References:
        [1] Author. (Year). Title. Journal.
    """
    
    def __init__(
        self,
        param1: Type,
        param2: Type = default,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize component.
        
        Args:
            param1: Description
            param2: Description (default: default value)
            config: Optional configuration
        """
        super().__init__()
        
        # Store parameters
        self.param1 = param1
        self.param2 = param2
        self.config = config or {}
        
        # Initialize internal state
        self._internal_state = None
    
    # Implement required methods here
    # (see section 4.2-4.5)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(param1={self.param1}, param2={self.param2})"
```

### 5.2 Type Specifications

**Input types:**
- Scalar: `float`, `int`
- Vector: `NDArray` with shape `(n,)`
- Matrix: `NDArray` with shape `(n, m)`
- Structured: Custom types for special structures

**Parameter dictionaries:**
```python
# Standard format for params
params = {
    'param_name': value,  # float or NDArray
    'hyperparameter': value,
}
```

**Return types:**
- Log-density: `float` or `NDArray`
- Gradient: `NDArray`
- Hessian: `NDArray` or `sparse.spmatrix` or `StructuredMatrix`
- Precision matrix: `sparse.spmatrix` (preferred) or `StructuredMatrix`

### 5.3 Parameter Constraints

**Input validation:**
```python
def _validate_params(self, params: Dict[str, Any]) -> None:
    """Validate parameter dictionary."""
    # Check required keys
    required = ['param1', 'param2']
    for key in required:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
    
    # Check value constraints
    if params['tau'] <= 0:
        raise ValueError("Precision tau must be positive")
    
    # Check dimensions
    # ...
```

**Constraints to enforce:**
- Positivity: $\tau > 0$, $\sigma > 0$
- Bounds: $0 < \rho < 1$ for correlations
- Dimension compatibility: 
- Other domain restrictions: 

### 5.4 Numerical Stability Considerations

**Common issues and solutions:**

1. **Log-domain computation:**
   - Never compute `exp(log_p)` if not necessary
   - Use `logsumexp` for sums of probabilities
   - Return log-density, not density

2. **Overflow/Underflow:**
   ```python
   # Bad
   p = exp(large_number)
   
   # Good
   log_p = large_number  # keep in log domain
   ```

3. **Matrix conditioning:**
   - Add jitter to precision matrices if needed: `Q + epsilon * I`
   - Use Cholesky decomposition carefully
   - Check condition numbers

4. **Gradient computation:**
   - Use numerically stable formulas
   - Avoid cancellation errors
   - Consider automatic differentiation

### 5.5 Special Implementation Details

**For Models:**
- Sparsity pattern representation
- Efficient precision matrix construction
- Constraint handling

**For Likelihoods:**
- Robust link function evaluation
- Numerical stability for extreme η values
- Efficient deviance calculation

**For Priors:**
- Log-domain throughout
- Handle improper priors carefully
- Reparametrization for optimization

---

## 6. Backend Integration

### 6.1 Matrix Type Requirements

**Precision matrix structure (for models):**
- [ ] Dense: Full matrix representation
- [ ] Sparse: Specify pattern
  - Sparsity pattern: 
  - Non-zeros per row: 
  - Format: CSR / CSC / COO
- [ ] Structured: Specify type
  - Type: Banded / Toeplitz / Block / etc.
  - Parameters: 

**Matrix construction:**
```python
def precision_matrix(self, hyperparams):
    """Construct precision matrix."""
    if self._structure == 'sparse':
        # Use scipy.sparse
        from scipy.sparse import diags, csr_matrix
        # Construct efficiently
        Q = ...
        return Q
    elif self._structure == 'banded':
        # Use structured matrix
        from backend.datastructures.matrix import BandedMatrix
        Q = BandedMatrix(...)
        return Q
```

### 6.2 Required Linear Algebra Operations

**Operations needed:**
| Operation | Usage | Backend Component |
|-----------|-------|-------------------|
| Matrix-vector product | | |
| Solve linear system | | |
| Cholesky factorization | | |
| Determinant (log) | | |
| Selected inversion | | |

**Solver requirements:**
- Solver type: 
- Factorization: 
- Special properties: 

### 6.3 Computational Complexity

**Per-method complexity:**
| Method | Time | Space | Notes |
|--------|------|-------|-------|
| `log_density` | O() | O() | |
| `gradient` | O() | O() | |
| `hessian` | O() | O() | |
| `precision_matrix` | O() | O() | |

**Bottlenecks:**
- 

**Optimization opportunities:**
- 

---

## 7. DALIA Integration

### 7.1 Usage Within DALIA

**DALIA modules that use this component:**
1. **Module:** `dalia.core.inla`
   - **Purpose:** 
   - **Methods called:** 
   
2. **Module:** `dalia.mode_finding`
   - **Purpose:** 
   - **Methods called:** 

3. **Module:** `dalia.integration`
   - **Purpose:** 
   - **Methods called:** 

### 7.2 Data Flow in DALIA

**Typical workflow:**
```
1. User specifies model using this component
   ↓
2. DALIA.core extracts model structure
   ↓
3. DALIA.mode_finding calls gradient/hessian
   ↓
4. DALIA.integration uses precision_matrix
   ↓
5. Results processed by post_model_fitting
```

### 7.3 Interface with Other Statistical Components

**Component dependencies:**
- **Depends on:** (e.g., prior depends on hyperparameters)
- **Used by:** (e.g., likelihood used by INLA)
- **Interacts with:** (e.g., model + likelihood combination)

**Composition examples:**
```python
# Model + Likelihood combination
from statistical_modeling_toolbox.models import AR1Model
from statistical_modeling_toolbox.likelihoods import GaussianLikelihood

model = AR1Model(order=1)
likelihood = GaussianLikelihood(link='identity')

# DALIA combines them
dalia = DALIA(model=model, likelihood=likelihood, data=data)
```

### 7.4 Reparametrization in DALIA Context

**When does reparametrization occur?**
- During optimization: 
- During integration: 
- For hyperparameters: 

**Transformation handling:**
- Who handles transformation? (Component vs DALIA): 
- Jacobian adjustments: 
- Back-transformation for output: 

---

## 8. Validation Strategy

### 8.1 Correctness Validation

#### Test 1: Known Analytical Results
```python
def test_analytical_case():
    """Test against known analytical result."""
    # Setup special case with known solution
    component = ComponentName(...)
    
    # Known result
    expected_log_p = ...  # from theory
    
    # Compute
    result = component.log_density(x, params)
    
    # Verify
    np.testing.assert_allclose(result, expected_log_p, rtol=1e-10)
```

**Analytical test cases:**
- Case 1: 
- Case 2: 

#### Test 2: Cross-Validation with External Libraries
```python
def test_against_scipy():
    """Compare with SciPy implementation."""
    from scipy.stats import distribution_name
    
    component = ComponentName(...)
    scipy_dist = scipy.stats.distribution_name(...)
    
    x = np.random.randn(100)
    
    our_result = component.log_density(x, params)
    scipy_result = scipy_dist.logpdf(x)
    
    np.testing.assert_allclose(our_result, scipy_result, rtol=1e-8)
```

**External library comparisons:**
- SciPy: 
- R-INLA: 
- Stan: 
- Other: 

#### Test 3: Internal Consistency
```python
def test_normalization():
    """Test that distribution normalizes correctly."""
    # Numerical integration should give 1 (or log(1) = 0)
    pass

def test_moments_consistency():
    """Test analytical vs empirical moments."""
    # Sample and compare with analytical moments
    pass
```

### 8.2 Gradient Validation

#### Finite Difference Check
```python
def test_gradient_finite_difference():
    """Validate gradient using finite differences."""
    component = ComponentName(...)
    
    x = np.random.randn(10)
    params = {'param1': 1.0, 'param2': 0.5}
    
    # Analytical gradient
    grad_analytical = component.gradient(x, params, wrt='x')
    
    # Numerical gradient
    eps = 1e-7
    grad_numerical = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        
        grad_numerical[i] = (
            component.log_density(x_plus, params) - 
            component.log_density(x_minus, params)
        ) / (2 * eps)
    
    # Compare
    np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-5)
```

**Tolerance specifications:**
- Relative tolerance: 1e-5 to 1e-7
- Absolute tolerance: 1e-10
- Step size: 1e-7 to 1e-8

#### Hessian Validation
```python
def test_hessian_finite_difference():
    """Validate Hessian using finite differences of gradient."""
    # Similar to gradient check but for second derivatives
    pass
```

### 8.3 Special Cases and Edge Cases

**Edge cases to test:**
- [ ] Zero values: x = 0
- [ ] Extreme values: x → ±∞
- [ ] Boundary values: At domain boundaries
- [ ] Degenerate parameters: τ → 0, τ → ∞
- [ ] High dimensions: n large
- [ ] Single observation: n = 1

**Example tests:**
```python
def test_edge_case_zero():
    """Test behavior at x = 0."""
    pass

def test_edge_case_extreme_precision():
    """Test with very high/low precision."""
    pass
```

### 8.4 Integration Tests

**Integration with DALIA:**
```python
def test_dalia_integration():
    """Test full integration with DALIA workflow."""
    from dalia.core import DALIA
    
    # Setup
    component = ComponentName(...)
    dalia = DALIA(model=component, ...)
    
    # Run DALIA workflow
    dalia.fit()
    
    # Verify results make sense
    assert dalia.converged
    # ...
```

### 8.5 Numerical Accuracy Requirements

**Tolerance levels:**
- Log-density: rtol=1e-10, atol=1e-12
- Gradient: rtol=1e-6, atol=1e-8
- Hessian: rtol=1e-4, atol=1e-6

**Justification:**
- Higher precision for log-density (used in mode finding)
- Moderate precision for gradients (optimization tolerance)
- Lower precision acceptable for Hessian (second-order approximation)

---

## 9. Usage Examples

### 9.1 Standalone Usage
```python
"""
Basic standalone usage of the component.
"""
import numpy as np
from statistical_modeling_toolbox.category import ComponentName

# Create component
component = ComponentName(param1=value1, param2=value2)

# Evaluate log-density
x = np.array([1.0, 2.0, 3.0])
params = {'param_a': 0.0, 'param_b': 1.0}

log_p = component.log_density(x, params)
print(f"Log-density: {log_p}")

# Compute gradient
grad = component.gradient(x, params, wrt='x')
print(f"Gradient: {grad}")

# For models: get precision matrix
if hasattr(component, 'precision_matrix'):
    Q = component.precision_matrix(hyperparams={'tau': 1.0})
    print(f"Precision matrix shape: {Q.shape}")
    print(f"Sparsity: {Q.nnz / Q.size * 100:.2f}%")
```

### 9.2 Integration with DALIA
```python
"""
Complete DALIA workflow using this component.
"""
from dalia.core import DALIA, DALIAConfig
from statistical_modeling_toolbox.models import ComponentName
from statistical_modeling_toolbox.likelihoods import GaussianLikelihood
from statistical_modeling_toolbox.priors import LogGammaPrior

# Setup model
model = ComponentName(param1=value1)
model.build(data={'locations': locations})

# Setup likelihood
likelihood = GaussianLikelihood(link='identity')

# Setup priors for hyperparameters
priors = {
    'tau': LogGammaPrior(a=1.0, b=0.001),
}

# Configure DALIA
config = DALIAConfig(
    mode_finding={'method': 'newton', 'tolerance': 1e-6},
    integration={'method': 'gaussian_quadrature', 'n_points': 21},
)

# Create and fit DALIA
dalia = DALIA(
    model=model,
    likelihood=likelihood,
    priors=priors,
    data=data,
    config=config
)

# Fit
results = dalia.fit()

# Access results
print(f"Posterior mean: {results.posterior_mean}")
print(f"Posterior std: {results.posterior_std}")
print(f"Hyperparameters: {results.hyperparameters}")
```

### 9.3 Simulation Study
```python
"""
Simulation study to validate the component.
"""
import numpy as np
import matplotlib.pyplot as plt

# True parameters
true_params = {'param': true_value}

# Generate synthetic data
component = ComponentName(...)
n_samples = 1000
data = component.sample(n_samples, true_params, random_state=42)

# Fit using DALIA
# ... (as above)

# Compare estimates with truth
# ...

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
# Plot theoretical density
x_range = np.linspace(data.min(), data.max(), 100)
log_p = component.log_density(x_range, true_params)
plt.plot(x_range, np.exp(log_p), label='True density')
plt.legend()
plt.title('Data vs True Distribution')

plt.subplot(1, 2, 2)
# Plot posterior distribution of parameters
# ...
plt.title('Posterior Distribution')

plt.tight_layout()
plt.show()
```

### 9.4 Performance Testing
```python
"""
Test computational performance.
"""
import time
import numpy as np

component = ComponentName(...)

# Test different problem sizes
sizes = [100, 1000, 10000, 100000]
times = []

for n in sizes:
    x = np.random.randn(n)
    params = {'param': 1.0}
    
    # Time log_density
    start = time.time()
    for _ in range(10):  # average over 10 runs
        _ = component.log_density(x, params)
    elapsed = (time.time() - start) / 10
    times.append(elapsed)
    
    print(f"n={n:6d}: {elapsed*1000:6.2f} ms")

# Plot scaling
import matplotlib.pyplot as plt
plt.loglog(sizes, times, 'o-')
plt.xlabel('Problem size (n)')
plt.ylabel('Time (s)')
plt.title('Computational Scaling')
plt.grid(True)
plt.show()
```

---

## 10. Performance Considerations

### 10.1 Computational Bottlenecks
*Identified hotspots:*
1. 
2. 

*Profiling results:*


### 10.2 Optimization Strategies
**Algorithmic optimizations:**
- 

**Implementation optimizations:**
- Vectorization: 
- Caching: 
- JIT compilation: 
- C/Cython extensions: 

### 10.3 Memory Usage
**Memory footprint:**
- For problem size n: O()
- Main allocations: 

**Memory optimization:**
- In-place operations: 
- Avoid copies: 

### 10.4 Scalability
**Scaling characteristics:**
- Small problems (n < 1000): 
- Medium problems (n = 1000-10000): 
- Large problems (n > 10000): 

**Parallel opportunities:**
- 

---

## 11. Implementation Roadmap

### 11.1 Phase 1: Core Implementation
**Tasks:**
- [ ] Implement basic class structure
- [ ] Implement log_density method
- [ ] Implement gradient method
- [ ] Implement hessian method
- [ ] Basic unit tests

**Estimated effort:** 

**Deliverables:**
- Working implementation
- Unit tests passing
- Basic documentation

### 11.2 Phase 2: Advanced Features
**Tasks:**
- [ ] Implement model-specific methods (if applicable)
- [ ] Implement likelihood-specific methods (if applicable)
- [ ] Optimization and performance tuning
- [ ] Extensive testing

**Estimated effort:** 

**Deliverables:**
- Complete feature set
- Optimized code
- Comprehensive tests

### 11.3 Phase 3: Integration and Validation
**Tasks:**
- [ ] Integrate with DALIA
- [ ] Integration tests
- [ ] Validation studies
- [ ] Documentation and examples

**Estimated effort:** 

**Deliverables:**
- Full DALIA integration
- Validation report
- User documentation

---

## 12. Known Limitations

### 12.1 Theoretical Limitations
*When does this component fail or perform poorly?*


*Asymptotic behavior:*


### 12.2 Numerical Limitations
*Numerical stability issues:*


*Precision limitations:*


### 12.3 Computational Limitations
*Scalability limits:*


*Performance bottlenecks:*


---

## 13. Future Extensions

### 13.1 Planned Improvements
- [ ] Feature 1: Priority: High/Medium/Low
- [ ] Feature 2: Priority: High/Medium/Low

### 13.2 Research Directions
*Theoretical extensions:*


*Methodological improvements:*


---

## 14. References and Related Work

### 14.1 Primary Literature
1. 
2. 

### 14.2 INLA References
1. Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. Chapman & Hall/CRC.
2. 

### 14.3 Software References
*Related implementations:*
- R-INLA: 
- PyMC: 
- Stan: 
- Other: 

### 14.4 Comparison with Other Implementations
| Feature | This Implementation | R-INLA | PyMC | Stan |
|---------|-------------------|--------|------|------|
| | | | | |

---

## 15. Appendices

### Appendix A: Mathematical Derivations
*Detailed derivations of key formulas:*

**Gradient derivation:**
$$
\frac{\partial}{\partial x} \log p(x | \theta) = 
$$

**Hessian derivation:**
$$
\frac{\partial^2}{\partial x^2} \log p(x | \theta) = 
$$

**Precision matrix structure (for models):**


### Appendix B: Alternative Parametrizations
*Document different parametrization schemes:*


### Appendix C: Numerical Experiments
*Record of validation experiments:*


### Appendix D: Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| | | Initial specification |

---

## Change Log
| Date | Author | Version | Changes |
|------|--------|---------|---------|
| | | 0.1 | Initial specification |

---

## Sign-off
| Role | Name | Date | Signature |
|------|------|------|-----------|
| Specification Author | | | |
| Statistical Reviewer | | | |
| DALIA Lead | | | |
