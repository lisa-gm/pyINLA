# Copyright 2024-2025 DALIA authors. All rights reserved.
from dalia import NDArray
from scipy.sparse import spmatrix
from dalia import sp, xp

import numpy as np

from dalia.configs.priorhyperparameters_config import (
    GammaPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class GammaPriorHyperparameters(PriorHyperparameters):
    """Gamma prior hyperparameters.

    p(theta) = (beta^alpha / Gamma(alpha)) * theta^(alpha - 1) * exp(-beta * theta)

    and in log scale:
    log p(theta) = alpha * log(beta) - log(Gamma(alpha)) + (alpha - 1) * log(theta) - beta * theta

    where theta is typically a positive parameter such as a precision or rate.

    Parameters
    ----------
    config : GammaPriorHyperparametersConfig
        Configuration object containing alpha and beta parameters.

    Attributes
    ----------
    alpha : float. alpha > 0
        Shape parameter of the Gamma distribution.
    beta : float. beta > 0
        Rate parameter of the Gamma distribution.
    normalizing_constant : float
        Precomputed normalizing constant for log probability evaluation.
    """

    def __init__(
        self,
        config: GammaPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Gamma prior hyperparameters.

        Parameters
        ----------
        config : GammaPriorHyperparametersConfig
            Configuration containing alpha (shape) and beta (rate) parameters.

        Raises
        ------
        ValueError
            If alpha or beta are not positive.
        """
        super().__init__(config)

        self.alpha: float = config.alpha
        self.beta: float = config.beta

        # Validate alpha and beta are positive
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"Beta must be positive, got {self.beta}")

        self.log_normalizing_constant: float = self.alpha * xp.log(self.beta) - float(
            sp.special.gammaln(self.alpha)
        )

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Transform between external and internal parameter representations.

        The Gamma distribution is defined for positive values, but optimization
        often works better in unconstrained space. This method transforms
        between theta (positive) and log(theta) (unconstrained).

        Parameters
        ----------
        theta : float or NDArray
            Parameter value(s) to transform.
        direction : str
            Transformation direction:
            - "forward": theta -> log(theta) (external to internal)
            - "backward": log(theta) -> theta (internal to external)

        Returns
        -------
        float or NDArray
            Transformed parameter value(s).

        Raises
        ------
        ValueError
            If direction is not "forward" or "backward", "forward_jacobian" or "backward_log_jacobian".
        """

        if direction == "forward":  # input is theta (external)
            theta_scaled = xp.log(theta)
        elif direction == "backward":  # input is theta (internal)
            theta_scaled = xp.exp(theta)
        elif direction == "forward_jacobian":  # input is theta (external)
            theta_scaled = 1 / theta  #
        elif direction == "backward_log_jacobian":  # input is theta (internal)
            theta_scaled = (
                theta  # log |d(exp(theta))/d(theta)| = log |exp(theta)| = theta
            )
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the prior probability density.

        Computes the probability density of the Gamma distribution
        at the given theta value in its external/user representation.

        Is mainly be used for plotting to understand the shape of the prior.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the prior.
            Must be positive (external/user representation).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Prior probability density at theta.

        Notes
        -----
        The computation follows:
            p(θ) = C * θ^(α - 1) * exp(-β * θ)
        where C is the normalizing constant, α is the shape parameter,
        and β is the rate parameter.
        """

        prior = (
            xp.exp(self.log_normalizing_constant)
            * theta ** (self.alpha - 1)
            * xp.exp(-self.beta * theta)
        )

        return prior

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Computes the log probability density of the Gamma distribution
        at the given theta value in its external/user representation.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the log prior.
            Must be positive (external/user representation).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at theta.

        Notes
        -----
        The computation follows:
            log p(θ) = C + (α - 1) * log(θ) - β * θ
        where C is the normalizing constant, α is the shape parameter,
        and β is the rate parameter.
        """

        if theta <= 0:
            raise ValueError(f"Theta must be positive for Gamma prior, got {theta}")

        log_prior = (
            self.log_normalizing_constant
            + (self.alpha - 1) * xp.log(theta)
            - self.beta * theta
        )

        return log_prior

    def evaluate_internal_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density transformed to internal space.

        Computes the log prior in unconstrained (internal/log) space by applying
        a log-Jacobian correction for the change of variables from constrained
        external space (θ > 0) to unconstrained internal space (θ_internal ∈ ℝ).

        Parameters
        ----------
        theta : float
            Parameter value in INTERNAL representation (R).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density in internal space:
            log p(θ_internal) = log p_external(g^(-1)(θ_internal)) + log|dθ/dθ_internal g^(-1)(θ_internal)|

        Notes
        -----
        The transformation uses the change of variables that is computed using
        rescale_hyperparameters_to_internal() with the "backward_log_jacobian"
        direction to account for the Jacobian of the transformation.
        """

        theta_external = self.rescale_hyperparameters_to_internal(theta, "backward")

        transformed_log_prior = self.evaluate_log_prior(
            theta_external
        ) + self.rescale_hyperparameters_to_internal(theta, "backward_log_jacobian")

        return transformed_log_prior


if __name__ == "__main__":
    """
    Test Gaussian quadrature for functions using rescale_hyperparameters_to_internal() 
    from gamma prior hyperparameters.
    
    We start with normally distributed random variables in internal space (unconstrained)
    that get reparametrized to external space (positive) using the gamma prior's 
    rescaling function.
    """

    from dalia.utils.gaussian_quadrature import compute_variance_gauss_hermite

    print("=" * 80)
    print("Testing Gaussian Quadrature with Gamma Prior Rescaling")
    print("=" * 80)

    # Create a inverse gamma prior configuration
    alpha_values = [1.0, 3.0, 5.0]
    beta_values = [0.5, 1.0, 2.0]

    for alpha, beta in zip(alpha_values, beta_values):
        print(f"\nTesting alpha={alpha}, beta={beta}")
        config = GammaPriorHyperparametersConfig(alpha=alpha, beta=beta)
        gamma_prior = GammaPriorHyperparameters(config=config)

        ## compare against scipy implementation
        from scipy.stats import gamma

        ## test that the log prior evaluation matches scipy's implementation for a range of theta values in external space
        test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        print("Comparing log prior evaluations with scipy.stats.gamma:")
        for val in test_values:
            p_dalia = gamma_prior.evaluate_prior(val)
            ## note: scipy's gamma takes scale = 1/beta
            p_scipy = gamma.pdf(val, a=alpha, scale=1 / beta)
            print(
                f"  θ = {val:4.1f}: DALIA p = {p_dalia:.6f}, "
                f"scipy p = {p_scipy:.6f}, diff = {abs(p_dalia - p_scipy):.2e}"
            )
            if abs(p_dalia - p_scipy) > 1e-6:
                raise ValueError(
                    "Prior evaluation does not match scipy implementation."
                )

        for val in test_values:
            log_p_dalia = gamma_prior.evaluate_log_prior(val)
            log_p_scipy = gamma.logpdf(val, a=alpha, scale=1 / beta)
            print(
                f"  θ = {val:4.1f}: DALIA log p = {log_p_dalia:.6f}, "
                f"scipy log p = {log_p_scipy:.6f}, diff = {abs(log_p_dalia - log_p_scipy):.2e}"
            )
            if abs(log_p_dalia - log_p_scipy) > 1e-6:
                raise ValueError(
                    "Log prior evaluation does not match scipy implementation."
                )

        # test that the transformed log prior matches the expected value
        # compare using sampling
        N = 1000000
        theta_external_samples = np.random.gamma(shape=alpha, scale=1 / beta, size=N)
        theta_internal_samples = np.log(
            theta_external_samples
        )  # Forward transformation: y = g(x)

        # xmin, xmax = theta_external_samples.min(), theta_external_samples.max()
        ymin, ymax = (
            theta_internal_samples.min(),
            theta_internal_samples.max(),
        )  # np.log(xmin), np.log(xmax)
        counts, bins = np.histogram(
            theta_internal_samples, bins=500, range=(ymin, ymax), density=True
        )

        y_grid = np.linspace(ymin, ymax, 1000)

        # Find the center of each histogram bin for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Filter out bins with 0 counts to prevent np.log(0) -> -inf errors
        valid_bins = counts > 0
        empirical_log_density = np.log(counts[valid_bins])
        filtered_bin_centers = bin_centers[valid_bins]

        # 4. Compute Theoretical Log-Density Curve
        theoretical_log_density = gamma_prior.evaluate_internal_log_prior(y_grid)

        from matplotlib import pyplot as plt

        # 5. Plot Direct Log-Space Comparison
        plt.figure(figsize=(9, 6))

        # Plot empirical log-density data points
        plt.scatter(
            filtered_bin_centers,
            empirical_log_density,
            color="royalblue",
            s=15,
            alpha=0.8,
            label="Empirical Log-Densities (from samples)",
        )

        # Plot your theoretical log-density function
        plt.plot(
            y_grid,
            theoretical_log_density,
            color="crimson",
            lw=2.5,
            label="your_implemented_log_f_Y(y)",
        )

        plt.title("Direct Log-Space Validation ($\ln(f_Y(y))$)", fontsize=14)
        plt.xlabel("y", fontsize=12)
        plt.ylabel("Log-Density", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)

        plt.show()

    ######### Check forward transformation
    # Suppose I have now fitted a Gaussian in internal space and want to check that the
    # forward transformation to external space gives the correct density shape.
    # Do this by sampling from the Gaussian in internal space, transforming to external space using the gamma
    # prior's rescaling function,
    # and comparing the empirical density of the transformed distribution to the empirical density of the

    # Create gamma prior configuration
    mean_values = [-1.0, 3.0, 5.0]
    sd_values = [0.5, 1.0, 2.0]

    for mean, sd in zip(mean_values, sd_values):
        N = 1000000
        internal_samples = np.random.normal(loc=mean, scale=sd, size=N)
        external_samples = np.exp(internal_samples)  # Forward transformation: y = g(x)

        xmin, xmax = external_samples.min(), external_samples.max()
        counts, bins = np.histogram(
            external_samples, bins=500, range=(xmin, xmax), density=True
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2

        valid_bins = counts > 0
        empirical_log_density = np.log(counts[valid_bins])
        filtered_bin_centers = bin_centers[valid_bins]

        external_grid = np.linspace(xmin, xmax, 1000)

        from scipy.stats import norm

        theoretical_density = norm.pdf(
            gamma_prior.rescale_hyperparameters_to_internal(external_grid, "forward"),
            loc=mean,
            scale=sd,
        ) * np.abs(
            gamma_prior.rescale_hyperparameters_to_internal(
                external_grid, "forward_jacobian"
            )
        )

        plt.figure(figsize=(9, 6))
        plt.scatter(
            bin_centers,
            counts,
            color="royalblue",
            s=15,
            alpha=0.8,
            label="Empirical Densities (from samples)",
        )
        plt.plot(
            external_grid,
            theoretical_density,
            color="crimson",
            lw=2.5,
            label="Theoretical Density with Jacobian Correction",
        )
        plt.title(
            "Validation of Forward Projection with Jacobian Correction", fontsize=14
        )
        plt.xlabel("x (External Space)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.show()

    print("\nAll value direct value comparisons passed. Check sampling plots.")
