"""
[Prior Name] - Hyperparameter Prior Specification
==================================================

Component Type: Prior (Hyperparameter Distribution)

Statistical Definition
----------------------
[Brief description of what this prior represents]

[Typical use cases: hyperparameter regularization, weakly informative, informative, etc.]


Mathematical form:

.. math::
    % For prior: \pi(\theta) = p(\theta | hyperparameters)
    % Example for log-gamma: \log(\tau) \sim Gamma(a, b)
    % Example for PC prior: P(range < u) = \alpha


Prior structure:
    Parameter: θ (hyperparameter being given a prior)
    Hyperparameters: [a, b, ...] (parameters of the prior distribution)
    Support: θ ∈ [domain]

Hyperparameters:
    - a: [interpretation, e.g., shape parameter]
    - b: [interpretation, e.g., rate parameter]


Properties
----------
Support (domain):
    θ ∈ [domain, e.g., (0, ∞), ℝ, (0, 1)]

Statistical properties:
    Mode: [if available]
    Mean: [if available]
    Variance: [if available]

Prior type:
    [ ] Conjugate prior
    [ ] Non-informative (flat, improper)
    [ ] Weakly informative
    [ ] Informative
    [ ] Penalized Complexity (PC) prior

Tail behavior:
    [Light-tailed, heavy-tailed, exponential decay, etc.]


Use Cases
---------
When to use this prior:
    - For hyperparameter: [precision τ, correlation ρ, range parameter, etc.]
    - Information level: [when you have/don't have prior knowledge]
    -

Typical applications:
    - Precision parameters (inverse variance) in GMRF models
    - Correlation parameters in temporal/spatial models
    - Range/smoothness parameters in SPDE models
    - Regularization in high-dimensional problems

When NOT to use:
    -
    -

Alternatives:
    [Other priors to consider and when to prefer them]
    - Uniform: For non-informative approach
    - Half-Cauchy: For heavy-tailed regularization
    - PC prior: For principled weakly informative priors


Parametrization
---------------
Natural parametrization:
    [Parameters as naturally expressed]
    θ in natural space

INLA/Optimization parametrization:
    [How parameters are expressed for computation]
    ψ = T(θ) in working space

    Common transformations:
    - Log: ψ = log(θ) for θ > 0
    - Logit: ψ = log(θ/(1-θ)) for θ ∈ (0,1)
    - Identity: ψ = θ

Transformation formulas:
    Forward: ψ = T(θ) =
    Inverse: θ = T⁻¹(ψ) =
    Jacobian: |dθ/dψ| =

Working space rationale:
    [Why this transformation? Unconstrained optimization, numerical stability, etc.]


Required Methods
----------------
Core Methods (all priors must implement):
    [x] log_density(theta, hyperparams) - Log-prior density
    [x] gradient(theta, hyperparams, wrt='theta') - Gradient
    [x] hessian(theta, hyperparams, wrt='theta') - Hessian

Parametrization methods:
    [x] to_working_scale(theta) - Transform to ψ
    [x] from_working_scale(psi) - Transform to θ
    [x] jacobian_log_det(theta) - Log |dθ/dψ|

Optional:
    [ ] sample(n, hyperparams) - Generate samples
    [ ] mode(hyperparams) - Prior mode
    [ ] default_hyperparameters() - Sensible defaults

Method signatures:
    See implementation section below


Computational Considerations
-----------------------------
Computational complexity:
    log_density(): O(1) [typically very simple]
    gradient(): O(1)
    hessian(): O(1)

Numerical stability:
    [Working scale prevents overflow/underflow]
    [Boundary behavior near 0 or 1]

Special considerations:
    [Improper priors, normalization constants]


DALIA Integration
-----------------
Used by DALIA modules:
    - dalia.core.inla: Assigns priors to hyperparameters
    - dalia.model_fitting: Uses gradient/hessian for hyperparameter optimization
    - dalia.integration: Integrates over hyperparameters in Laplace approximation
    - dalia.post_model_fitting: Computes posterior for hyperparameters

What DALIA expects:
    - log_density(θ) for prior evaluation
    - Gradient and Hessian for optimization in working scale
    - Proper Jacobian adjustment when transforming scales

Hyperparameter assignment:
    Each model hyperparameter can be assigned a prior:
    .. code-block:: python

        priors = {
            'tau': LogGammaPrior(a=1, b=0.001),
            'rho': BetaPrior(a=1, b=1),
            'kappa': PCPrior(u=0.5, alpha=0.05)
        }

Transformation handling:
    [DALIA optimizes in working scale ψ but evaluates models in natural scale θ]


Validation
----------
Correctness checks:
    1. Normalization (if proper prior):
       - ∫ π(θ) dθ = 1 (numerical integration check)

    2. Transformation consistency:
       - θ = T⁻¹(T(θ)) (round-trip test)
       - Jacobian matches numerical derivative

    3. Gradient validation:
       - Finite difference check with tolerance rtol=1e-5
       - Including Jacobian term in working scale

Numerical tests:
    - Boundary behavior (θ → 0, θ → ∞)
    - Working scale stability
    - Gradient accuracy across parameter range

Cross-validation:
    Compare with: [R-INLA, Stan priors, scipy.stats, etc.]

Test cases:
    1. Standard values: [typical hyperparameter values]
    2. Edge cases: [boundaries, extreme values]
    3. Against reference implementation


Example Usage
-------------
.. code-block:: python

    # Basic standalone usage
    from statistical_modeling_toolbox.priors import PriorName
    import numpy as np

    # Create prior with hyperparameters
    prior = PriorName(a=1.0, b=0.001)

    # Evaluate log-prior for hyperparameter value
    theta = 2.5  # e.g., precision parameter
    log_prior = prior.log_density(theta)
    print(f"Log-prior at θ={theta}: {log_prior:.4f}")

    # Work in optimization scale
    psi = prior.to_working_scale(theta)
    log_prior_psi = prior.log_density_working_scale(psi)

    # Gradient for optimization
    grad = prior.gradient(theta, wrt='theta')
    grad_psi = prior.gradient_working_scale(psi)

    # Sample from prior
    samples = prior.sample(n=1000, random_state=42)
    print(f"Prior mean: {samples.mean():.4f}")

    # Integration with DALIA
    from dalia.core import DALIA
    from statistical_modeling_toolbox.models import AR1Model
    from statistical_modeling_toolbox.likelihoods import PoissonLikelihood
    from statistical_modeling_toolbox.priors import LogGammaPrior, BetaPrior

    # Setup model
    model = AR1Model(order=1).build(data={'time': time_points})
    likelihood = PoissonLikelihood(link='log')

    # Assign priors to hyperparameters
    priors = {
        'tau': LogGammaPrior(a=1.0, b=0.001),     # precision
        'rho': BetaPrior(a=2.0, b=2.0)            # correlation
    }

    # Fit with DALIA
    dalia = DALIA(
        model=model,
        likelihood=likelihood,
        priors=priors,
        data=data
    )
    results = dalia.fit()

    # Access posterior for hyperparameters
    print(f"Posterior precision: {results.hyperparams['tau']}")
    print(f"Posterior correlation: {results.hyperparams['rho']}")


Implementation Notes
--------------------
Working scale transformation:
    [Why this specific transformation]
    [Numerical advantages]
    [Boundary handling]

Jacobian computation:
    [Analytical formula]
    [When to include/exclude in log-density]

Special cases:
    [Improper priors: b → 0 for log-gamma]
    [Boundary behavior]
    [Numerical issues near boundaries]


Notes & Open Questions
----------------------
Implementation notes:
    -

Mathematical properties:
    -

Numerical considerations:
    -

Open questions:
    -


References
----------
.. [1] Rue, H., & Held, L. (2005). Gaussian Markov Random Fields. Chapman & Hall/CRC.
       Chapter on hyperparameter priors.
.. [2] Simpson, D., et al. (2017). Penalising Model Component Complexity:
       A Principled, Practical Approach to Constructing Priors.
       Statistical Science, 32(1), 1-28.
.. [3] [Prior-specific reference]


Related Priors
--------------
See Also:
    [List related priors in the toolbox]
    - SimilarPrior: For comparison
    - AlternativePrior: Different information level


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
[ ] log_density implemented
[ ] Gradient/Hessian implemented
[ ] Working scale transformation implemented
[ ] Jacobian computed correctly
[ ] Gradient validation passed
[ ] Integration with DALIA tested
[ ] Edge cases handled
[ ] Documentation complete
[ ] Ready for production

"""

# Standard library imports
from typing import Dict, Any, Optional, Union, Tuple, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import gammaln, loggamma

# Local imports - Base class
from statistical_modeling_toolbox.priors.prior import Prior

# Module exports
__all__ = [
    "PriorName",
    "PriorConfig",
]

# Module-level constants
EPSILON = 1e-10  # Small value to prevent numerical issues


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PriorConfig:
    """
    Configuration for [PriorName].

    Attributes
    ----------
    validate_inputs : bool
        Whether to validate inputs, default: True
    use_working_scale : bool
        Whether to work in transformed scale, default: True
    numerical_tolerance : float
        Tolerance for numerical operations, default: 1e-10
    """

    validate_inputs: bool = True
    use_working_scale: bool = True
    numerical_tolerance: float = EPSILON


# ============================================================================
# Main Implementation
# ============================================================================


class PriorName(Prior):
    """
    [Brief one-line description of the prior]

    [Detailed description explaining:
    - What hyperparameter this is typically used for
    - Level of informativeness
    - When to use it
    - How it integrates with DALIA]

    Mathematical Form
    -----------------
    .. math::
        \\pi(\\theta | a, b) = [formula]

    For example, for log-gamma prior:
    .. math::
        \\log(\\tau) \\sim \\text{Gamma}(a, b)

    which gives:
    .. math::
        \\pi(\\tau) = \\frac{b^a}{\\Gamma(a)} \\tau^{-1} (\\log \\tau)^{a-1}
                     \\exp(-b \\log \\tau)

    Parameters
    ----------
    a : float
        [Interpretation, e.g., shape parameter]
    b : float
        [Interpretation, e.g., rate parameter]
    config : Optional[PriorConfig], default=None
        Configuration object

    Attributes
    ----------
    a : float
        Shape/first parameter
    b : float
        Rate/second parameter
    is_proper : bool
        Whether the prior is proper (integrates to 1)

    Examples
    --------
    >>> prior = PriorName(a=1.0, b=0.001)
    >>> log_p = prior.log_density(theta=2.5)
    >>> grad = prior.gradient(theta=2.5)

    >>> # Work in optimization scale
    >>> psi = prior.to_working_scale(theta=2.5)
    >>> log_p_psi = prior.log_density_working_scale(psi)

    Notes
    -----
    [Important notes about:]
    - Support and parameter constraints
    - Informativeness (vague, weakly informative, informative)
    - Typical hyperparameter ranges
    - Relationship to other priors

    References
    ----------
    .. [1] Primary reference for this prior.
    .. [2] INLA reference if applicable.

    See Also
    --------
    RelatedPrior : Alternative prior for same purpose
    """

    def __init__(
        self, a: float, b: float, config: Optional[PriorConfig] = None
    ) -> None:
        """
        Initialize [PriorName].

        Parameters
        ----------
        a : float
            [Description and valid range]
        b : float
            [Description and valid range]
        config : Optional[PriorConfig], default=None
            Configuration object

        Raises
        ------
        ValueError
            If hyperparameters are invalid
        """
        super().__init__()

        # Configuration
        if config is None:
            config = PriorConfig()
        self.config = config

        # Validate hyperparameters
        if config.validate_inputs:
            self._validate_hyperparams(a, b)

        # Store hyperparameters
        self.a = a
        self.b = b

        # Properties
        self.is_proper = True  # Set to False if improper (e.g., b=0)
        if b <= 0:
            self.is_proper = False

    def log_density(
        self, theta: Union[float, NDArray], include_jacobian: bool = False
    ) -> Union[float, NDArray]:
        """
        Evaluate log-prior density.

        Computes: log π(θ)

        Parameters
        ----------
        theta : float or NDArray
            Hyperparameter value(s) in natural scale
        include_jacobian : bool, default=False
            Whether to include Jacobian term for transformation
            Set to True when working in transformed scale

        Returns
        -------
        log_p : float or NDArray
            Log-prior density value(s)

        Raises
        ------
        ValueError
            If theta is out of valid range

        Notes
        -----
        For priors on transformed parameters, the Jacobian adjustment is:
        log π(ψ) = log π(θ(ψ)) + log |dθ/dψ|

        When include_jacobian=True, this method returns log π(θ) + log |dθ/dψ|.

        Examples
        --------
        >>> log_p = prior.log_density(theta=2.5)
        >>> # For optimization in working scale:
        >>> psi = prior.to_working_scale(2.5)
        >>> log_p_psi = prior.log_density(
        ...     prior.from_working_scale(psi),
        ...     include_jacobian=True
        ... )
        """
        if self.config.validate_inputs:
            self._validate_theta(theta)

        theta = np.asarray(theta)

        # Compute log-prior in natural scale
        # IMPLEMENT ACTUAL FORMULA
        # Example for log-gamma prior on precision τ:
        # log_tau = np.log(theta)
        # log_p = (self.a - 1) * log_tau - self.b * log_tau - np.log(theta) - gammaln(self.a) + self.a * np.log(self.b)

        log_p = 0.0  # PLACEHOLDER - IMPLEMENT

        # Add Jacobian if requested
        if include_jacobian:
            log_jac = self.jacobian_log_det(theta)
            log_p = log_p + log_jac

        # If input is vector, sum for total log-prior
        if np.ndim(log_p) > 0:
            return np.sum(log_p)
        return float(log_p)

    def gradient(
        self, theta: Union[float, NDArray], wrt: Literal["theta", "working"] = "theta"
    ) -> Union[float, NDArray]:
        """
        Compute gradient of log-prior.

        Parameters
        ----------
        theta : float or NDArray
            Hyperparameter value(s) in natural scale
        wrt : {'theta', 'working'}, default='theta'
            Compute gradient with respect to:
            - 'theta': gradient w.r.t. natural parameter
            - 'working': gradient w.r.t. working scale ψ

        Returns
        -------
        gradient : float or NDArray
            Gradient vector, same shape as theta

        Notes
        -----
        For optimization, DALIA needs gradient in working scale.
        This is computed using chain rule:

        ∂log π/∂ψ = (∂log π/∂θ) × (∂θ/∂ψ)

        When include_jacobian=True in log_density, also need:
        ∂log π/∂ψ = ∂log π/∂θ × ∂θ/∂ψ + ∂log|J|/∂ψ

        Examples
        --------
        >>> grad_theta = prior.gradient(theta=2.5, wrt='theta')
        >>> grad_psi = prior.gradient(theta=2.5, wrt='working')
        """
        theta = np.asarray(theta)

        if wrt == "theta":
            # Gradient w.r.t. natural parameter
            # IMPLEMENT: ∂log π(θ) / ∂θ
            grad = 0.0  # PLACEHOLDER

        elif wrt == "working":
            # Gradient w.r.t. working scale (chain rule)
            grad_theta = self.gradient(theta, wrt="theta")
            dtheta_dpsi = self._derivative_from_working_scale(theta)
            grad = grad_theta * dtheta_dpsi

            # Add Jacobian gradient if using transformed scale
            grad_jac = self._gradient_jacobian_log_det(theta)
            grad = grad + grad_jac

        else:
            raise ValueError(f"wrt must be 'theta' or 'working', got {wrt}")

        return grad

    def hessian(
        self, theta: Union[float, NDArray], wrt: Literal["theta", "working"] = "theta"
    ) -> Union[float, NDArray]:
        """
        Compute Hessian (second derivative) of log-prior.

        Parameters
        ----------
        theta : float or NDArray
            Hyperparameter value(s)
        wrt : {'theta', 'working'}, default='theta'
            Compute Hessian with respect to:
            - 'theta': Hessian w.r.t. natural parameter
            - 'working': Hessian w.r.t. working scale

        Returns
        -------
        hessian : float or NDArray
            Hessian (scalar or diagonal for independent parameters)

        Notes
        -----
        For scalar hyperparameters, returns the second derivative.
        For vector hyperparameters (rare), returns diagonal elements.

        Examples
        --------
        >>> H = prior.hessian(theta=2.5, wrt='working')
        """
        theta = np.asarray(theta)

        if wrt == "theta":
            # Second derivative w.r.t. θ
            # IMPLEMENT: ∂²log π(θ) / ∂θ²
            hess = 0.0  # PLACEHOLDER

        elif wrt == "working":
            # Hessian w.r.t. working scale (chain rule, more complex)
            hess_theta = self.hessian(theta, wrt="theta")
            grad_theta = self.gradient(theta, wrt="theta")

            dtheta_dpsi = self._derivative_from_working_scale(theta)
            d2theta_dpsi2 = self._second_derivative_from_working_scale(theta)

            # Apply chain rule for second derivative
            hess = hess_theta * (dtheta_dpsi**2) + grad_theta * d2theta_dpsi2

            # Add Jacobian Hessian contribution
            hess_jac = self._hessian_jacobian_log_det(theta)
            hess = hess + hess_jac

        else:
            raise ValueError(f"wrt must be 'theta' or 'working', got {wrt}")

        return hess

    # ========================================================================
    # Working scale transformations
    # ========================================================================

    def to_working_scale(self, theta: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Transform from natural scale to working scale: ψ = T(θ).

        Parameters
        ----------
        theta : float or NDArray
            Parameter in natural scale

        Returns
        -------
        psi : float or NDArray
            Parameter in working scale (unconstrained)

        Notes
        -----
        Common transformations:
        - Log: ψ = log(θ) for θ > 0
        - Logit: ψ = log(θ/(1-θ)) for θ ∈ (0,1)
        - Identity: ψ = θ for θ ∈ ℝ

        Working scale should be unconstrained (ψ ∈ ℝ) for optimization.

        Examples
        --------
        >>> psi = prior.to_working_scale(theta=2.5)
        >>> # For log transform: psi = log(2.5) ≈ 0.916
        """
        theta = np.asarray(theta)

        # IMPLEMENT transformation
        # Example for positive parameters: return np.log(theta)
        # Example for (0,1) parameters: return np.log(theta / (1 - theta))
        # Example for unbounded: return theta

        psi = theta  # PLACEHOLDER - IMPLEMENT
        return psi

    def from_working_scale(self, psi: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Transform from working scale to natural scale: θ = T⁻¹(ψ).

        Parameters
        ----------
        psi : float or NDArray
            Parameter in working scale (unconstrained)

        Returns
        -------
        theta : float or NDArray
            Parameter in natural scale (possibly constrained)

        Notes
        -----
        Inverse transformations:
        - Exp: θ = exp(ψ) for log transform
        - Logistic: θ = 1/(1+exp(-ψ)) for logit transform
        - Identity: θ = ψ for identity transform

        Examples
        --------
        >>> theta = prior.from_working_scale(psi=0.916)
        >>> # For log transform: theta = exp(0.916) ≈ 2.5
        """
        psi = np.asarray(psi)

        # IMPLEMENT inverse transformation
        # Example for log: return np.exp(psi)
        # Example for logit: return 1.0 / (1.0 + np.exp(-psi))
        # Example for identity: return psi

        theta = psi  # PLACEHOLDER - IMPLEMENT
        return theta

    def jacobian_log_det(self, theta: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Compute log absolute Jacobian determinant: log |dθ/dψ|.

        Parameters
        ----------
        theta : float or NDArray
            Parameter in natural scale

        Returns
        -------
        log_jac : float or NDArray
            Log absolute Jacobian determinant

        Notes
        -----
        For transformations:
        - Log (ψ = log θ): log |dθ/dψ| = log(θ)
        - Logit (ψ = log(θ/(1-θ))): log |dθ/dψ| = log(θ(1-θ))
        - Identity: log |dθ/dψ| = 0

        This is needed to correctly account for the transformation:
        π(ψ) = π(θ(ψ)) |dθ/dψ|

        Examples
        --------
        >>> log_jac = prior.jacobian_log_det(theta=2.5)
        >>> # For log transform: log_jac = log(2.5) ≈ 0.916
        """
        theta = np.asarray(theta)

        # IMPLEMENT Jacobian
        # Example for log: return np.log(theta)
        # Example for logit: return np.log(theta * (1 - theta))
        # Example for identity: return 0.0

        log_jac = 0.0  # PLACEHOLDER - IMPLEMENT
        return log_jac

    # ========================================================================
    # Convenience methods for working scale
    # ========================================================================

    def log_density_working_scale(
        self, psi: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        """
        Evaluate log-prior in working scale.

        Convenience method: log π(ψ) = log π(θ(ψ)) + log |dθ/dψ|

        Parameters
        ----------
        psi : float or NDArray
            Parameter in working scale

        Returns
        -------
        log_p : float or NDArray
            Log-prior density in working scale

        Examples
        --------
        >>> psi = prior.to_working_scale(2.5)
        >>> log_p = prior.log_density_working_scale(psi)
        """
        theta = self.from_working_scale(psi)
        return self.log_density(theta, include_jacobian=True)

    def gradient_working_scale(
        self, psi: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        """
        Compute gradient in working scale.

        Parameters
        ----------
        psi : float or NDArray
            Parameter in working scale

        Returns
        -------
        gradient : float or NDArray
            Gradient w.r.t. ψ
        """
        theta = self.from_working_scale(psi)
        return self.gradient(theta, wrt="working")

    def hessian_working_scale(
        self, psi: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        """
        Compute Hessian in working scale.

        Parameters
        ----------
        psi : float or NDArray
            Parameter in working scale

        Returns
        -------
        hessian : float or NDArray
            Hessian w.r.t. ψ
        """
        theta = self.from_working_scale(psi)
        return self.hessian(theta, wrt="working")

    # ========================================================================
    # Optional methods
    # ========================================================================

    def sample(self, n: int, random_state: Optional[int] = None) -> NDArray:
        """
        Generate random samples from the prior.

        Parameters
        ----------
        n : int
            Number of samples
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        samples : NDArray, shape (n,)
            Random samples from π(θ)

        Examples
        --------
        >>> samples = prior.sample(n=1000, random_state=42)
        >>> print(f"Prior mean: {samples.mean():.4f}")
        >>> print(f"Prior std: {samples.std():.4f}")
        """
        if random_state is not None:
            np.random.seed(random_state)

        # IMPLEMENT using np.random or scipy.stats
        # Example for log-gamma:
        # log_samples = np.random.gamma(self.a, 1/self.b, size=n)
        # return np.exp(log_samples)

        raise NotImplementedError("sample not implemented")

    def mode(self) -> float:
        """
        Compute the prior mode.

        Returns
        -------
        mode : float
            Value of θ that maximizes π(θ)

        Notes
        -----
        Useful for initialization in optimization.
        May not exist for all priors (e.g., uniform).

        Examples
        --------
        >>> mode = prior.mode()
        >>> print(f"Prior mode: {mode:.4f}")
        """
        # IMPLEMENT analytical mode if available
        # Example for log-gamma: mode = exp((a-1)/b)
        raise NotImplementedError("mode not implemented")

    def mean(self) -> float:
        """
        Compute the prior mean (if it exists).

        Returns
        -------
        mean : float
            E[θ]
        """
        # IMPLEMENT if analytical mean exists
        raise NotImplementedError("mean not implemented")

    def variance(self) -> float:
        """
        Compute the prior variance (if it exists).

        Returns
        -------
        variance : float
            Var[θ]
        """
        # IMPLEMENT if analytical variance exists
        raise NotImplementedError("variance not implemented")

    @staticmethod
    def default_hyperparameters() -> Dict[str, float]:
        """
        Return sensible default hyperparameters.

        Returns
        -------
        defaults : dict
            Dictionary with default values for a, b, etc.

        Notes
        -----
        These should represent a reasonable weakly informative prior
        for typical applications.

        Examples
        --------
        >>> defaults = PriorName.default_hyperparameters()
        >>> prior = PriorName(**defaults)
        """
        return {
            "a": 1.0,
            "b": 0.001,  # Vague prior
        }

    # ========================================================================
    # Internal helper methods
    # ========================================================================

    def _derivative_from_working_scale(self, theta: NDArray) -> NDArray:
        """
        Compute dθ/dψ.

        This is the derivative of the inverse transformation.
        """
        # IMPLEMENT based on transformation
        # Example for log: return theta
        # Example for logit: return theta * (1 - theta)
        return np.ones_like(theta)  # PLACEHOLDER

    def _second_derivative_from_working_scale(self, theta: NDArray) -> NDArray:
        """
        Compute d²θ/dψ².

        Needed for Hessian in working scale.
        """
        # IMPLEMENT based on transformation
        # Example for log: return theta
        # Example for logit: return theta * (1 - theta) * (1 - 2*theta)
        return np.zeros_like(theta)  # PLACEHOLDER

    def _gradient_jacobian_log_det(self, theta: NDArray) -> NDArray:
        """
        Compute gradient of log Jacobian: ∂log|J|/∂ψ.
        """
        # IMPLEMENT
        # Example for log: return 1.0
        # Example for logit: return (1 - 2*theta)
        return np.zeros_like(theta)  # PLACEHOLDER

    def _hessian_jacobian_log_det(self, theta: NDArray) -> NDArray:
        """
        Compute Hessian of log Jacobian: ∂²log|J|/∂ψ².
        """
        # IMPLEMENT
        return np.zeros_like(theta)  # PLACEHOLDER

    # ========================================================================
    # Validation methods
    # ========================================================================

    def _validate_hyperparams(self, a: float, b: float) -> None:
        """Validate prior hyperparameters."""
        # IMPLEMENT checks based on prior
        # Example: check a > 0, b >= 0, etc.
        if a <= 0:
            raise ValueError(f"Parameter 'a' must be positive, got {a}")
        if b < 0:
            raise ValueError(f"Parameter 'b' must be non-negative, got {b}")

    def _validate_theta(self, theta: Union[float, NDArray]) -> None:
        """Validate hyperparameter value."""
        theta = np.asarray(theta)

        # IMPLEMENT checks based on support
        # Example for positive parameters: check theta > 0
        # Example for (0,1) parameters: check 0 < theta < 1
        if np.any(theta <= 0):
            raise ValueError("Parameter θ must be positive")

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(a={self.a}, b={self.b})"


# ============================================================================
# Helper Functions
# ============================================================================


def validate_transformation_roundtrip(
    prior: PriorName, theta_test: NDArray, tol: float = 1e-10
) -> bool:
    """
    Validate transformation round-trip: θ = T⁻¹(T(θ)).

    Parameters
    ----------
    prior : PriorName
        Prior instance to test
    theta_test : NDArray
        Test values for parameter (in valid range)
    tol : float, default=1e-10
        Tolerance for comparison

    Returns
    -------
    passed : bool
        True if validation passed

    Examples
    --------
    >>> prior = PriorName(a=1.0, b=0.001)
    >>> theta_test = np.array([0.1, 1.0, 5.0, 10.0])
    >>> assert validate_transformation_roundtrip(prior, theta_test)
    """
    psi = prior.to_working_scale(theta_test)
    theta_recovered = prior.from_working_scale(psi)

    return np.allclose(theta_test, theta_recovered, rtol=tol, atol=tol)


def validate_jacobian_numerical(
    prior: PriorName, theta_test: NDArray, eps: float = 1e-7, rtol: float = 1e-5
) -> bool:
    """
    Validate Jacobian using numerical derivatives.

    Parameters
    ----------
    prior : PriorName
        Prior instance
    theta_test : NDArray
        Test values
    eps : float, default=1e-7
        Finite difference step
    rtol : float, default=1e-5
        Relative tolerance

    Returns
    -------
    passed : bool
        True if validation passed
    """
    # Analytical Jacobian
    log_jac_analytical = prior.jacobian_log_det(theta_test)

    # Numerical Jacobian: d(log|dθ/dψ|)/dψ ≈ (log|J(θ+ε)| - log|J(θ-ε)|) / 2ε
    # This is complex, simplified check:
    psi = prior.to_working_scale(theta_test)

    # Check that dθ/dψ matches numerical derivative
    theta_plus = prior.from_working_scale(psi + eps)
    theta_minus = prior.from_working_scale(psi - eps)
    dtheta_dpsi_numerical = (theta_plus - theta_minus) / (2 * eps)
    dtheta_dpsi_analytical = prior._derivative_from_working_scale(theta_test)

    return np.allclose(dtheta_dpsi_analytical, dtheta_dpsi_numerical, rtol=rtol)


def validate_gradient_finite_difference(
    prior: PriorName, theta_test: NDArray, eps: float = 1e-7, rtol: float = 1e-5
) -> bool:
    """
    Validate gradient using finite differences.

    Parameters
    ----------
    prior : PriorName
        Prior instance
    theta_test : NDArray
        Test values
    eps : float, default=1e-7
        Finite difference step
    rtol : float, default=1e-5
        Relative tolerance

    Returns
    -------
    passed : bool
        True if validation passed
    """
    # Analytical gradient
    grad_analytical = prior.gradient(theta_test, wrt="theta")

    # Numerical gradient
    grad_numerical = np.zeros_like(theta_test)
    for i in range(len(theta_test)):
        theta_plus = theta_test.copy()
        theta_plus[i] += eps
        theta_minus = theta_test.copy()
        theta_minus[i] -= eps

        grad_numerical[i] = (
            prior.log_density(theta_plus) - prior.log_density(theta_minus)
        ) / (2 * eps)

    return np.allclose(grad_analytical, grad_numerical, rtol=rtol)


# ============================================================================
# End of module
# ============================================================================
