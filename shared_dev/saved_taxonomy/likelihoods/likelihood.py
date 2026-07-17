"""
[Likelihood Name] - Statistical Likelihood Specification
=========================================================

Component Type: Likelihood (Observation Model)

Statistical Definition
----------------------
[Brief description of what this likelihood represents]

[Typical use cases: count data, continuous data, binary outcomes, etc.]


Mathematical form:

.. math::
    % For likelihood: y_i | \eta_i, \phi \sim Distribution(\mu_i, \phi)
    % where \mu_i = g^{-1}(\eta_i) and g is the link function
    % Example for Poisson: y_i | \eta_i \sim Poisson(e^{\eta_i})


Likelihood structure:
    Observation: y (response variable)
    Linear predictor: η = X β + Z u (from model)
    Expected value: μ = g^{-1}(η) (inverse link function)
    Distribution: y | μ, φ ~ [Distribution family]

Parameters:
    - mu (μ): Expected value / mean parameter
    - phi (φ): Dispersion parameter (if applicable)

Link function:
    Canonical: [e.g., log for Poisson, logit for Binomial]
    Alternative links: [identity, log, logit, probit, etc.]


Properties
----------
Distributional family:
    [Exponential family / Non-exponential family]
    [Discrete / Continuous]

Support (domain):
    y ∈ [domain, e.g., {0, 1, 2, ...} for Poisson, [0, ∞) for Gamma]

Mean-variance relationship:
    E[Y] = μ
    Var[Y] = [variance function, e.g., μ for Poisson, μ(1-μ) for Binomial]

Dispersion:
    [ ] No dispersion parameter (e.g., Poisson, Binomial)
    [ ] Has dispersion parameter φ (e.g., Gaussian, Negative Binomial)

Canonical link:
    η = g(μ) = [canonical link formula]
    Properties of canonical link: [simplified sufficient statistics, etc.]


Use Cases
---------
When to use this likelihood:
    - Data type: [count, continuous, binary, proportion, etc.]
    - Data characteristics: [bounded, non-negative, overdispersed, etc.]
    -

Typical applications:
    -
    -

When NOT to use:
    -
    -

Alternatives:
    [Other likelihoods to consider and when to prefer them]


Link Functions
--------------
Canonical link:
    Name: [e.g., log, logit, identity]
    Formula: η = g(μ) =
    Inverse: μ = g^{-1}(η) =

    Properties:
        - Natural parameter equals linear predictor
        - Simplifies Fisher information
        - [Other properties]

Alternative links:
    1. [Link name]:
       - η = g(μ) =
       - μ = g^{-1}(η) =
       - When to use:

    2. [Another link]:
       - η = g(μ) =
       - μ = g^{-1}(η) =
       - When to use:

Link function properties:
    - Domain: μ ∈ [domain from distribution]
    - Range: η ∈ [range, usually ℝ]
    - Monotonicity: [increasing/decreasing]
    - Boundary behavior: [as μ → 0, ∞, etc.]


Required Methods
----------------
Core Methods (all likelihoods must implement):
    [x] log_density(y, mu, phi) - Log-likelihood evaluation
    [x] gradient(y, mu, phi, wrt) - Gradient computation
    [x] hessian(y, mu, phi, wrt) - Hessian computation
    [x] link_function(mu) - Apply link η = g(μ)
    [x] inverse_link_function(eta) - Apply inverse μ = g^{-1}(η)
    [x] deviance(y_obs, y_pred) - Deviance for model comparison

Optional but recommended:
    [ ] derivative_inverse_link(eta) - dμ/dη for chain rule
    [ ] variance_function(mu) - Variance as function of mean
    [ ] sample(n, mu, phi) - Generate observations
    [ ] initial_values(y) - Good starting values for optimization

Method signatures:
    See implementation section below


Parametrization
---------------
Natural parametrization:
    [Parameters as typically presented in statistics, e.g., λ for Poisson]


INLA/GLM parametrization:
    Linear predictor: η
    Mean parameter: μ = g^{-1}(η)
    Dispersion: φ (if applicable)

Working scale (for optimization):
    [Log-scale or other transformation for φ if needed]


Computational Considerations
-----------------------------
Computational complexity:
    log_density(): O(n) [n = number of observations]
    gradient(): O(n)
    hessian(): O(n) [usually diagonal or simple structure]

Numerical stability:
    [Issues with extreme μ values, overflow/underflow concerns]
    [How to handle boundary cases]

Special considerations:
    [Large counts, zero-inflation, perfect separation, etc.]


DALIA Integration
-----------------
Used by DALIA modules:
    - dalia.core.inla: Combines with model for full Bayesian inference
    - dalia.mode_finding: Uses gradient/hessian for finding posterior mode
    - dalia.integration: Uses Laplace approximation around mode
    - dalia.post_model_fitting: Deviance for DIC, WAIC computation

What DALIA expects:
    - log_density(y, mu) where μ comes from inverse link of linear predictor
    - Gradient and Hessian w.r.t. η (chain rule from μ)
    - Proper handling of dispersion parameters

Combination with model:
    Full model: y | η, φ ~ Likelihood(g^{-1}(η), φ)
                η = X β + Z u
                u | θ ~ Model(0, Q(θ)^{-1})

Data requirements:
    - Response variable y (observed data)
    - Optional: weights, offsets, trial sizes (for Binomial), etc.


Validation
----------
Correctness checks:
    1. Link function properties:
       - μ = g^{-1}(g(μ)) (round-trip test)
       - Domain and range constraints satisfied

    2. Gradient validation:
       - Finite difference check with tolerance rtol=1e-5
       - Chain rule: ∂log p/∂η = (∂log p/∂μ)(∂μ/∂η)

    3. Known special cases:
       - Specific parameter values with analytical solutions
       - Edge cases (y=0, large y, etc.)

Numerical tests:
    - Stability of link functions at boundaries
    - Overflow/underflow in exponentials
    - Cancellation in deviance computations

Cross-validation:
    Compare with: [R glm(), statsmodels, scipy.stats, etc.]

Test cases:
    1. Simple data: [e.g., y=[0,1,2], standard parameters]
    2. Edge cases: [zeros, large values, boundaries]
    3. Against reference implementation


Example Usage
-------------
.. code-block:: python

    # Basic standalone usage
    from statistical_modeling_toolbox.likelihoods import LikelihoodName
    import numpy as np

    # Create likelihood with canonical link
    likelihood = LikelihoodName(link='canonical')

    # Observed data and linear predictor
    y = np.array([0, 1, 3, 2, 5])  # observations
    eta = np.array([-0.5, 0.0, 1.0, 0.5, 1.5])  # linear predictor

    # Convert to mean scale
    mu = likelihood.inverse_link_function(eta)

    # Evaluate log-likelihood
    log_lik = likelihood.log_density(y, mu, phi=1.0)
    print(f"Log-likelihood: {log_lik}")

    # Compute gradient w.r.t. eta (for optimization)
    grad = likelihood.gradient(y, mu, phi=1.0, wrt='eta')

    # Deviance for model comparison
    y_pred = mu  # predicted values
    dev = likelihood.deviance(y, y_pred)
    print(f"Deviance: {dev}")

    # Integration with DALIA
    from dalia.core import DALIA
    from statistical_modeling_toolbox.models import AR1Model

    # Setup
    model = AR1Model(order=1).build(data={'time': time_points})
    likelihood = LikelihoodName(link='log')

    # Fit with DALIA
    dalia = DALIA(
        model=model,
        likelihood=likelihood,
        data={'y': observations, 'time': time_points},
        formula='y ~ f(time, model=AR1())'
    )
    results = dalia.fit()

    # Access results
    print(f"Linear predictor: {results.eta}")
    print(f"Predicted mean: {results.mu}")
    print(f"DIC: {results.dic}")


Implementation Notes
--------------------
Link function implementation:
    [Specific numerical tricks for stability]
    [Handling extreme values]
    [Clamping if needed]

Gradient computation:
    [Use of chain rule]
    [Automatic differentiation vs analytical]

Deviance formula:
    [Specific formula for this distribution]
    [Saturated model definition]

Special cases:
    [How to handle zero observations, missing data, etc.]


Notes & Open Questions
----------------------
Implementation notes:
    -

Numerical considerations:
    -

Open questions:
    -


References
----------
.. [1] McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models.
       Chapman & Hall/CRC.
.. [2] [Likelihood-specific reference]
.. [3] [Additional references]


Related Likelihoods
-------------------
See Also:
    [List related likelihoods in the toolbox]
    - SimilarLikelihood: For comparison
    - ExtensionLikelihood: Generalization


Authors
-------
- [Author Name] <email@domain.com> ([Date])


Version History
---------------
- 0.1.0 ([Date]): Initial specification


Status
------
[ ] Specification complete
[ ] Link functions implemented and tested
[ ] log_density implemented
[ ] Gradient/Hessian implemented
[ ] Deviance implemented
[ ] Gradient validation passed
[ ] Cross-validation with reference implementation
[ ] Integration with DALIA tested
[ ] Edge cases handled
[ ] Documentation complete
[ ] Ready for production

"""

from dataclasses import dataclass
from enum import Enum

# Standard library imports
from typing import Any, Dict, Literal, Optional, Tuple, Union

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# Local imports - Base class
from statistical_modeling_toolbox.likelihoods.likelihood import Likelihood

# Module exports
__all__ = [
    "LikelihoodName",
    "LikelihoodConfig",
    "LinkFunction",
]

# Module-level constants
DEFAULT_DISPERSION = 1.0
EPSILON = 1e-10  # Small value to prevent log(0), division by zero


# ============================================================================
# Link Function Enum
# ============================================================================


class LinkFunction(str, Enum):
    """Available link functions for this likelihood."""

    CANONICAL = "canonical"  # e.g., 'log' for Poisson
    IDENTITY = "identity"
    LOG = "log"
    LOGIT = "logit"
    # Add others as needed: PROBIT, CLOGLOG, INVERSE, etc.


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class LikelihoodConfig:
    """
    Configuration for [LikelihoodName].

    Attributes
    ----------
    validate_inputs : bool
        Whether to validate inputs, default: True
    numerical_tolerance : float
        Tolerance for numerical operations, default: 1e-10
    clip_predictions : bool
        Whether to clip predictions to valid range, default: True
    """

    validate_inputs: bool = True
    numerical_tolerance: float = EPSILON
    clip_predictions: bool = True


# ============================================================================
# Main Implementation
# ============================================================================


class LikelihoodName(Likelihood):
    """
    [Brief one-line description of the likelihood]

    [Detailed description explaining:
    - What type of data this is appropriate for
    - Key properties (discrete/continuous, bounded, etc.)
    - When to use it
    - How it integrates with DALIA/GLM framework]

    Mathematical Form
    -----------------
    .. math::
        y_i | \\mu_i, \\phi \\sim [Distribution](\\mu_i, \\phi)

    where μ_i = g^{-1}(η_i) and η_i is the linear predictor.

    For example, for Poisson with log link:
    .. math::
        y_i | \\eta_i \\sim \\text{Poisson}(\\lambda_i = e^{\\eta_i})

    Parameters
    ----------
    link : str or LinkFunction, default='canonical'
        Link function to use:
        - 'canonical': [e.g., log for Poisson, logit for Binomial]
        - 'identity': μ = η
        - 'log': μ = exp(η)
        - 'logit': μ = 1/(1+exp(-η))
    dispersion : float, optional
        Dispersion parameter (if applicable), default: 1.0
    config : Optional[LikelihoodConfig], default=None
        Configuration object

    Attributes
    ----------
    link : LinkFunction
        Selected link function
    dispersion : float
        Dispersion parameter
    has_dispersion : bool
        Whether this likelihood has a dispersion parameter

    Examples
    --------
    >>> likelihood = LikelihoodName(link='canonical')
    >>> y = np.array([0, 1, 2, 3])
    >>> mu = np.array([0.5, 1.0, 2.0, 3.5])
    >>> log_lik = likelihood.log_density(y, mu)

    >>> # With link function
    >>> eta = np.array([-0.5, 0.0, 0.7, 1.2])
    >>> mu = likelihood.inverse_link_function(eta)
    >>> log_lik = likelihood.log_density(y, mu)

    Notes
    -----
    [Important notes about:]
    - Valid ranges for parameters
    - Numerical stability issues
    - Special cases (zero-inflation, perfect separation, etc.)
    - Computational efficiency

    References
    ----------
    .. [1] McCullagh & Nelder (1989). GLM.
    .. [2] Likelihood-specific reference.

    See Also
    --------
    RelatedLikelihood : Related observation model
    """

    def __init__(
        self,
        link: Union[str, LinkFunction] = "canonical",
        dispersion: Optional[float] = None,
        config: Optional[LikelihoodConfig] = None,
    ) -> None:
        """
        Initialize [LikelihoodName].

        Parameters
        ----------
        link : str or LinkFunction, default='canonical'
            Link function to use
        dispersion : float, optional
            Dispersion parameter (if applicable)
        config : Optional[LikelihoodConfig], default=None
            Configuration object

        Raises
        ------
        ValueError
            If link function is not supported
        """
        super().__init__()

        # Configuration
        if config is None:
            config = LikelihoodConfig()
        self.config = config

        # Link function
        if isinstance(link, str):
            try:
                self.link = LinkFunction(link)
            except ValueError:
                raise ValueError(f"Unsupported link function: {link}")
        else:
            self.link = link

        # Dispersion parameter
        self.has_dispersion = True  # Set to False for Poisson, Binomial
        if dispersion is None:
            self.dispersion = DEFAULT_DISPERSION
        else:
            if config.validate_inputs and dispersion <= 0:
                raise ValueError("Dispersion must be positive")
            self.dispersion = dispersion

    def log_density(
        self,
        y: Union[float, NDArray],
        mu: Union[float, NDArray],
        phi: Optional[float] = None,
    ) -> Union[float, NDArray]:
        """
        Evaluate log-likelihood for observations.

        Computes: log p(y | μ, φ)

        Parameters
        ----------
        y : float or NDArray
            Observed data
        mu : float or NDArray
            Mean parameter (expected value), same shape as y
            Must be μ = g^{-1}(η), i.e., on the mean scale, not linear predictor
        phi : float, optional
            Dispersion parameter (if applicable)
            If None, uses self.dispersion

        Returns
        -------
        log_lik : float or NDArray
            Log-likelihood value(s)
            If y and mu are vectors, returns sum over observations

        Raises
        ------
        ValueError
            If y or mu are out of valid range

        Notes
        -----
        This method computes the log-likelihood on the mean scale.
        For use with linear predictor η, first apply inverse link:
        μ = g^{-1}(η), then call log_density(y, μ).

        Examples
        --------
        >>> log_lik = likelihood.log_density(y=[0, 1, 2], mu=[0.5, 1.0, 2.0])

        >>> # With linear predictor
        >>> eta = np.array([-0.5, 0.0, 0.7])
        >>> mu = likelihood.inverse_link_function(eta)
        >>> log_lik = likelihood.log_density(y=[0, 1, 2], mu=mu)
        """
        # Handle dispersion
        if phi is None:
            phi = self.dispersion

        # Validate inputs
        if self.config.validate_inputs:
            self._validate_observations(y)
            self._validate_mean_parameter(mu)

        # Convert to arrays for vectorization
        y = np.asarray(y)
        mu = np.asarray(mu)

        # Clip mu to valid range if configured
        if self.config.clip_predictions:
            mu = self._clip_mu(mu)

        # Compute log-likelihood
        # IMPLEMENT ACTUAL FORMULA
        # Example for Poisson: log_lik = y * log(mu) - mu - log_factorial(y)
        # Example for Gaussian: log_lik = -0.5 * ((y - mu)**2 / phi + log(2*pi*phi))

        log_lik = 0.0  # PLACEHOLDER - IMPLEMENT

        # If inputs are vectors, return sum (total log-likelihood)
        if np.ndim(log_lik) > 0:
            return np.sum(log_lik)
        return float(log_lik)

    def gradient(
        self,
        y: Union[float, NDArray],
        mu: Union[float, NDArray],
        phi: Optional[float] = None,
        wrt: Literal["mu", "eta"] = "eta",
    ) -> Union[float, NDArray]:
        """
        Compute gradient of log-likelihood.

        Parameters
        ----------
        y : float or NDArray
            Observed data
        mu : float or NDArray
            Mean parameter
        phi : float, optional
            Dispersion parameter
        wrt : {'mu', 'eta'}, default='eta'
            Compute gradient with respect to:
            - 'mu': gradient w.r.t. mean parameter
            - 'eta': gradient w.r.t. linear predictor (uses chain rule)

        Returns
        -------
        gradient : float or NDArray
            Gradient vector, same shape as y

        Notes
        -----
        For optimization in DALIA, we need gradient w.r.t. η (linear predictor).
        This is computed using chain rule:

        ∂log p/∂η = (∂log p/∂μ) × (∂μ/∂η)

        where ∂μ/∂η = dg^{-1}(η)/dη depends on the link function.

        Examples
        --------
        >>> grad_mu = likelihood.gradient(y, mu, wrt='mu')
        >>> grad_eta = likelihood.gradient(y, mu, wrt='eta')
        """
        if phi is None:
            phi = self.dispersion

        # Convert to arrays
        y = np.asarray(y)
        mu = np.asarray(mu)

        if self.config.clip_predictions:
            mu = self._clip_mu(mu)

        if wrt == "mu":
            # Gradient w.r.t. mean parameter
            # IMPLEMENT: ∂log p(y | μ, φ) / ∂μ
            grad = 0.0  # PLACEHOLDER

        elif wrt == "eta":
            # Gradient w.r.t. linear predictor (chain rule)
            # ∂log p/∂η = (∂log p/∂μ) × (∂μ/∂η)

            grad_mu = self.gradient(y, mu, phi, wrt="mu")
            dmu_deta = self.derivative_inverse_link(mu)
            grad = grad_mu * dmu_deta

        else:
            raise ValueError(f"wrt must be 'mu' or 'eta', got {wrt}")

        return grad

    def hessian(
        self,
        y: Union[float, NDArray],
        mu: Union[float, NDArray],
        phi: Optional[float] = None,
        wrt: Literal["mu", "eta"] = "eta",
    ) -> Union[float, NDArray]:
        """
        Compute Hessian (second derivative) of log-likelihood.

        Parameters
        ----------
        y : float or NDArray
            Observed data
        mu : float or NDArray
            Mean parameter
        phi : float, optional
            Dispersion parameter
        wrt : {'mu', 'eta'}, default='eta'
            Compute Hessian with respect to:
            - 'mu': Hessian w.r.t. mean parameter
            - 'eta': Hessian w.r.t. linear predictor

        Returns
        -------
        hessian : float or NDArray
            Hessian (diagonal for independent observations)

        Notes
        -----
        For GLM with independent observations, Hessian is diagonal.
        Returns the diagonal elements as a vector.

        For wrt='eta', uses chain rule:
        ∂²log p/∂η² = (∂²log p/∂μ²)(∂μ/∂η)² + (∂log p/∂μ)(∂²μ/∂η²)

        Examples
        --------
        >>> H = likelihood.hessian(y, mu, wrt='eta')
        >>> # H contains diagonal elements of Hessian matrix
        """
        if phi is None:
            phi = self.dispersion

        y = np.asarray(y)
        mu = np.asarray(mu)

        if self.config.clip_predictions:
            mu = self._clip_mu(mu)

        if wrt == "mu":
            # Second derivative w.r.t. μ
            # IMPLEMENT: ∂²log p(y | μ, φ) / ∂μ²
            hess = 0.0  # PLACEHOLDER

        elif wrt == "eta":
            # Hessian w.r.t. η (chain rule)
            hess_mu = self.hessian(y, mu, phi, wrt="mu")
            grad_mu = self.gradient(y, mu, phi, wrt="mu")

            dmu_deta = self.derivative_inverse_link(mu)
            d2mu_deta2 = self.second_derivative_inverse_link(mu)

            hess = hess_mu * (dmu_deta**2) + grad_mu * d2mu_deta2

        else:
            raise ValueError(f"wrt must be 'mu' or 'eta', got {wrt}")

        return hess

    def link_function(self, mu: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Apply link function: η = g(μ).

        Maps mean parameter to linear predictor.

        Parameters
        ----------
        mu : float or NDArray
            Mean parameter (must be in valid range for distribution)

        Returns
        -------
        eta : float or NDArray
            Linear predictor, same shape as mu

        Raises
        ------
        ValueError
            If mu is out of valid range

        Notes
        -----
        The link function maps the constrained mean parameter space
        to the unconstrained linear predictor space.

        Examples
        --------
        >>> eta = likelihood.link_function(mu=2.5)
        >>> # For log link: eta = log(2.5) ≈ 0.916

        >>> # Vectorized
        >>> mu = np.array([0.5, 1.0, 2.0, 5.0])
        >>> eta = likelihood.link_function(mu)
        """
        mu = np.asarray(mu)

        if self.link == LinkFunction.IDENTITY:
            return mu

        elif self.link == LinkFunction.LOG:
            # Avoid log(0)
            mu_safe = np.maximum(mu, self.config.numerical_tolerance)
            return np.log(mu_safe)

        elif self.link == LinkFunction.LOGIT:
            # Avoid log(0) and log(∞)
            mu_safe = np.clip(
                mu, self.config.numerical_tolerance, 1 - self.config.numerical_tolerance
            )
            return np.log(mu_safe / (1 - mu_safe))

        elif self.link == LinkFunction.CANONICAL:
            # Map to specific canonical link for this likelihood
            # IMPLEMENT based on distribution
            return self.link_function(mu)  # Redirect to appropriate link

        else:
            raise NotImplementedError(f"Link function {self.link} not implemented")

    def inverse_link_function(
        self, eta: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        """
        Apply inverse link function: μ = g^{-1}(η).

        Maps linear predictor to mean parameter.

        Parameters
        ----------
        eta : float or NDArray
            Linear predictor (can be any real value)

        Returns
        -------
        mu : float or NDArray
            Mean parameter in valid range, same shape as eta

        Notes
        -----
        The inverse link ensures μ is in the valid range for the distribution
        (e.g., positive for Poisson, in [0,1] for Binomial).

        Examples
        --------
        >>> mu = likelihood.inverse_link_function(eta=1.0)
        >>> # For log link: mu = exp(1.0) ≈ 2.718

        >>> # Vectorized
        >>> eta = np.array([-1.0, 0.0, 1.0, 2.0])
        >>> mu = likelihood.inverse_link_function(eta)
        """
        eta = np.asarray(eta)

        if self.link == LinkFunction.IDENTITY:
            return eta

        elif self.link == LinkFunction.LOG:
            # Clip eta to prevent overflow in exp
            eta_safe = np.clip(eta, -20, 20)  # exp(20) ≈ 5e8
            return np.exp(eta_safe)

        elif self.link == LinkFunction.LOGIT:
            # Numerically stable logistic function
            # Use: 1/(1+exp(-x)) = exp(x)/(1+exp(x)) for x > 0
            eta_safe = np.clip(eta, -20, 20)
            return 1.0 / (1.0 + np.exp(-eta_safe))

        elif self.link == LinkFunction.CANONICAL:
            # Map to specific canonical link inverse
            # IMPLEMENT based on distribution
            return self.inverse_link_function(eta)

        else:
            raise NotImplementedError(f"Inverse link for {self.link} not implemented")

    def derivative_inverse_link(
        self, mu: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        """
        Compute derivative of inverse link function: dμ/dη.

        Parameters
        ----------
        mu : float or NDArray
            Mean parameter (output of inverse link)

        Returns
        -------
        derivative : float or NDArray
            dμ/dη evaluated at η such that μ = g^{-1}(η)

        Notes
        -----
        This is needed for the chain rule in gradient computation.
        We compute it as a function of μ rather than η for efficiency
        (we already have μ computed).

        For common links:
        - Identity: dμ/dη = 1
        - Log: dμ/dη = μ
        - Logit: dμ/dη = μ(1-μ)

        Examples
        --------
        >>> dmu = likelihood.derivative_inverse_link(mu=2.0)
        >>> # For log link: dμ/dη = μ = 2.0
        """
        mu = np.asarray(mu)

        if self.link == LinkFunction.IDENTITY:
            return np.ones_like(mu)

        elif self.link == LinkFunction.LOG:
            return mu

        elif self.link == LinkFunction.LOGIT:
            return mu * (1 - mu)

        elif self.link == LinkFunction.CANONICAL:
            # IMPLEMENT for canonical link
            return self.derivative_inverse_link(mu)

        else:
            raise NotImplementedError(f"Derivative for {self.link} not implemented")

    def second_derivative_inverse_link(
        self, mu: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        """
        Compute second derivative of inverse link: d²μ/dη².

        Parameters
        ----------
        mu : float or NDArray
            Mean parameter

        Returns
        -------
        second_derivative : float or NDArray
            d²μ/dη²

        Notes
        -----
        Needed for Hessian computation via chain rule.

        For common links:
        - Identity: d²μ/dη² = 0
        - Log: d²μ/dη² = μ
        - Logit: d²μ/dη² = μ(1-μ)(1-2μ)
        """
        mu = np.asarray(mu)

        if self.link == LinkFunction.IDENTITY:
            return np.zeros_like(mu)

        elif self.link == LinkFunction.LOG:
            return mu

        elif self.link == LinkFunction.LOGIT:
            return mu * (1 - mu) * (1 - 2 * mu)

        elif self.link == LinkFunction.CANONICAL:
            # IMPLEMENT
            return self.second_derivative_inverse_link(mu)

        else:
            raise NotImplementedError(
                f"Second derivative for {self.link} not implemented"
            )

    def deviance(
        self, y_obs: NDArray, y_pred: NDArray, phi: Optional[float] = None
    ) -> float:
        """
        Compute deviance for model comparison.

        Deviance = -2 * [log L(y; μ_fitted) - log L(y; μ_saturated)]

        where μ_saturated is the maximum likelihood estimate (perfect fit).

        Parameters
        ----------
        y_obs : NDArray
            Observed data
        y_pred : NDArray
            Predicted mean values (μ, not η)
        phi : float, optional
            Dispersion parameter

        Returns
        -------
        deviance : float
            Deviance value (lower is better, 0 is perfect fit)

        Notes
        -----
        Used in:
        - Model comparison (lower deviance = better fit)
        - DIC (Deviance Information Criterion)
        - WAIC (Watanabe-Akaike Information Criterion)

        For distributions in the exponential family, there's usually
        a simple formula not requiring log L evaluation.

        Examples
        --------
        >>> dev = likelihood.deviance(y_obs=y, y_pred=mu_fitted)
        >>> print(f"Model deviance: {dev:.2f}")
        """
        if phi is None:
            phi = self.dispersion

        y_obs = np.asarray(y_obs)
        y_pred = np.asarray(y_pred)

        if self.config.clip_predictions:
            y_pred = self._clip_mu(y_pred)

        # Compute deviance using distribution-specific formula
        # IMPLEMENT ACTUAL FORMULA
        # Example for Poisson:
        # dev = 2 * np.sum(y_obs * np.log(y_obs / y_pred) - (y_obs - y_pred))
        # (with special handling for y_obs=0)

        # Example for Gaussian:
        # dev = np.sum((y_obs - y_pred)**2) / phi

        deviance = 0.0  # PLACEHOLDER - IMPLEMENT

        return float(deviance)

    def variance_function(self, mu: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Variance as a function of mean: Var[Y] = V(μ).

        Parameters
        ----------
        mu : float or NDArray
            Mean parameter

        Returns
        -------
        variance : float or NDArray
            Variance, same shape as mu

        Notes
        -----
        For exponential family distributions:
        Var[Y] = φ × V(μ)

        where V(μ) is the variance function:
        - Gaussian: V(μ) = 1
        - Poisson: V(μ) = μ
        - Binomial: V(μ) = μ(1-μ)
        - Gamma: V(μ) = μ²

        Examples
        --------
        >>> var = likelihood.variance_function(mu=2.0)
        """
        # IMPLEMENT based on distribution
        # Example for Poisson: return mu
        # Example for Binomial: return mu * (1 - mu)
        raise NotImplementedError("variance_function not implemented")

    def sample(
        self,
        n: int,
        mu: Union[float, NDArray],
        phi: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> NDArray:
        """
        Generate random samples from the likelihood.

        Parameters
        ----------
        n : int
            Number of samples per mean value
        mu : float or NDArray
            Mean parameter(s)
        phi : float, optional
            Dispersion parameter
        random_state : int, optional
            Random seed

        Returns
        -------
        samples : NDArray
            Random samples
            If mu is scalar: shape (n,)
            If mu is vector of length m: shape (n, m)

        Examples
        --------
        >>> samples = likelihood.sample(n=100, mu=2.5, random_state=42)
        >>> print(f"Sample mean: {samples.mean():.2f}, expected: 2.5")
        """
        if random_state is not None:
            np.random.seed(random_state)

        if phi is None:
            phi = self.dispersion

        mu = np.asarray(mu)

        # IMPLEMENT using np.random
        # Example for Poisson: return np.random.poisson(mu, size=(n, *mu.shape))
        # Example for Gaussian: return np.random.normal(mu, np.sqrt(phi), size=(n, *mu.shape))

        raise NotImplementedError("sample not implemented")

    def initial_values(self, y: NDArray) -> NDArray:
        """
        Compute good initial values for linear predictor.

        Provides reasonable starting values η for optimization based on data.

        Parameters
        ----------
        y : NDArray
            Observed data

        Returns
        -------
        eta_init : NDArray
            Initial values for linear predictor

        Notes
        -----
        Common approaches:
        - Use link of observed values (with adjustments for zeros)
        - Use link of smoothed data
        - Use simple moments-based estimates

        Examples
        --------
        >>> eta_init = likelihood.initial_values(y=observations)
        """
        y = np.asarray(y)

        # Simple approach: use link of (y + small_constant)
        # Adjust based on specific likelihood
        mu_init = y + 0.1  # Prevent issues with zeros
        eta_init = self.link_function(mu_init)

        return eta_init

    # ========================================================================
    # Validation and utility methods
    # ========================================================================

    def _validate_observations(self, y: Union[float, NDArray]) -> None:
        """Validate observed data."""
        y = np.asarray(y)

        # IMPLEMENT distribution-specific checks
        # Example for Poisson: check non-negative integers
        # Example for Binomial: check 0 <= y <= n
        pass

    def _validate_mean_parameter(self, mu: Union[float, NDArray]) -> None:
        """Validate mean parameter."""
        mu = np.asarray(mu)

        # IMPLEMENT distribution-specific checks
        # Example for Poisson: check mu > 0
        # Example for Binomial: check 0 < mu < 1
        pass

    def _clip_mu(self, mu: NDArray) -> NDArray:
        """Clip mean parameter to valid range."""
        # IMPLEMENT based on distribution
        # Example for Poisson: return np.maximum(mu, EPSILON)
        # Example for Binomial: return np.clip(mu, EPSILON, 1 - EPSILON)
        return mu

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(link='{self.link.value}')"


# ============================================================================
# Helper Functions
# ============================================================================


def _log_factorial(n: Union[int, NDArray]) -> Union[float, NDArray]:
    """
    Compute log(n!) numerically stable.

    Uses Stirling approximation for large n.

    Parameters
    ----------
    n : int or NDArray
        Non-negative integer(s)

    Returns
    -------
    log_fact : float or NDArray
        log(n!)
    """
    from scipy.special import gammaln

    return gammaln(n + 1)


def validate_link_function_roundtrip(
    likelihood: LikelihoodName, mu_test: NDArray, tol: float = 1e-10
) -> bool:
    """
    Validate link function round-trip: μ = g^{-1}(g(μ)).

    Parameters
    ----------
    likelihood : LikelihoodName
        Likelihood instance to test
    mu_test : NDArray
        Test values for mean parameter (in valid range)
    tol : float, default=1e-10
        Tolerance for comparison

    Returns
    -------
    passed : bool
        True if validation passed

    Examples
    --------
    >>> likelihood = LikelihoodName(link='log')
    >>> mu_test = np.array([0.1, 1.0, 5.0, 10.0])
    >>> assert validate_link_function_roundtrip(likelihood, mu_test)
    """
    eta = likelihood.link_function(mu_test)
    mu_recovered = likelihood.inverse_link_function(eta)

    return np.allclose(mu_test, mu_recovered, rtol=tol, atol=tol)


def validate_gradient_finite_difference(
    likelihood: LikelihoodName,
    y: NDArray,
    mu: NDArray,
    eps: float = 1e-7,
    rtol: float = 1e-5,
) -> bool:
    """
    Validate gradient using finite differences.

    Parameters
    ----------
    likelihood : LikelihoodName
        Likelihood instance
    y : NDArray
        Test observations
    mu : NDArray
        Test mean parameters
    eps : float, default=1e-7
        Finite difference step size
    rtol : float, default=1e-5
        Relative tolerance

    Returns
    -------
    passed : bool
        True if validation passed
    """
    # Analytical gradient
    grad_analytical = likelihood.gradient(y, mu, wrt="mu")

    # Numerical gradient
    grad_numerical = np.zeros_like(mu)
    for i in range(len(mu)):
        mu_plus = mu.copy()
        mu_plus[i] += eps
        mu_minus = mu.copy()
        mu_minus[i] -= eps

        grad_numerical[i] = (
            likelihood.log_density(y[i], mu_plus[i])
            - likelihood.log_density(y[i], mu_minus[i])
        ) / (2 * eps)

    return np.allclose(grad_analytical, grad_numerical, rtol=rtol)


# ============================================================================
# End of module
# ============================================================================
