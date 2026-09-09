# Copyright 2024-2025 DALIA authors. All rights reserved.
from dalia import NDArray
from scipy.sparse import spmatrix
from dalia import sp, xp

import numpy as np

from dalia.configs.priorhyperparameters_config import (
    InverseGammaPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class InverseGammaPriorHyperparameters(PriorHyperparameters):
    """Inverse Gamma prior hyperparameters.

    p(theta) = (beta^alpha / Gamma(alpha)) * (1/theta)^(alpha + 1) * exp(-beta / theta)

    and in log scale:
    log p(theta) = alpha * log(beta) - log(Gamma(alpha)) - (alpha + 1) * log(theta) - beta / theta

    where theta is typically a positive parameter such as a variance.

    Internal parameterization uses log transformation:
    theta_internal = log(theta)
    theta_external = exp(theta_internal)

    Parameters
    ----------
    config : InverseGammaPriorHyperparametersConfig
        Configuration object containing alpha and beta parameters.

    Attributes
    ----------
    alpha : float. alpha > 0
        Shape parameter of the Inverse Gamma distribution.
    beta : float. beta > 0
        Scale parameter of the Inverse Gamma distribution.
    log_normalizing_constant : float
        Precomputed log of the normalizing constant for log probability evaluation.
    """

    def __init__(
        self,
        config: InverseGammaPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Inverse Gamma prior hyperparameters.

        Parameters
        ----------
        config : InverseGammaPriorHyperparametersConfig
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

        The Inverse Gamma distribution is defined for positive values (theta > 0), but
        optimization often works better in unconstrained space. This method transforms
        between theta (positive) and log(theta) (unconstrained).

        Parameters
        ----------
        theta : float or NDArray
            Parameter value(s) to transform.
        direction : str
            Transformation direction:
            - "forward": theta -> log(theta) (external to internal)
            - "backward": log(theta) -> theta (internal to external)
            - "forward_jacobian": Jacobian of forward transformation
            - "backward_log_jacobian": Log-Jacobian of backward transformation

        Returns
        -------
        float or NDArray
            Transformed parameter value(s).

        Raises
        ------
        ValueError
            If direction is not recognized.
        """
        if direction == "forward":  # input is theta (external)
            theta_scaled = -xp.log(theta)
        elif direction == "backward":  # input is theta (internal)
            theta_scaled = xp.exp(-theta)
        elif direction == "forward_jacobian":
            theta_scaled = 1.0 / xp.abs(theta)
        elif direction == "backward_log_jacobian":
            theta_scaled = -theta
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the prior probability density.

        Computes the probability density of the Inverse Gamma distribution
        at the given theta value in its external/user representation.

        Is mainly used for plotting to understand the shape of the prior.

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
            p(θ) = C * (1/θ)^(α + 1) * exp(-β / θ)
        where C is the normalizing constant, α is the shape parameter,
        and β is the scale parameter.
        """
        if xp.min(theta) <= 0:
            raise ValueError(f"Inverse Gamma theta must be positive. Got theta={theta}")

        prior = (
            xp.exp(self.log_normalizing_constant)
            * (1.0 / theta) ** (self.alpha + 1)
            * xp.exp(-self.beta / theta)
        )

        return prior

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Computes the log probability density of the Inverse Gamma distribution
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
            log p(θ) = C - (α + 1) * log(θ) - β / θ
        where C is the normalizing constant, α is the shape parameter,
        and β is the scale parameter.
        """

        if xp.min(theta) <= 0:
            raise ValueError(f"Inverse Gamma theta must be positive. Got theta={theta}")

        log_prior = (
            self.log_normalizing_constant
            - (self.alpha + 1) * xp.log(theta)
            - self.beta / theta
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
            Parameter value in INTERNAL representation (must be positive).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density in internal space:

        """

        theta_external = self.rescale_hyperparameters_to_internal(theta, "backward")

        transformed_log_prior = self.evaluate_log_prior(
            theta_external
        ) + self.rescale_hyperparameters_to_internal(theta, "backward_log_jacobian")

        return transformed_log_prior


if __name__ == "__main__":
    """
    Test Inverse Gamma prior hyperparameters with scipy validation.

    Validates:
    1. Prior evaluation against scipy.stats.invgamma
    2. Log-prior evaluation against scipy.stats.invgamma.logpdf
    3. Internal log-prior (transformed to log space) via empirical sampling
    """

    print("=" * 80)
    print("Testing Inverse Gamma Prior Hyperparameters")
    print("=" * 80)

    from scipy.stats import invgamma

    # Test configurations
    alpha_values = [1.0, 3.0, 5.0]
    beta_values = [0.5, 1.0, 2.0]

    for alpha, beta in zip(alpha_values, beta_values):
        print(f"\nTesting alpha={alpha}, beta={beta}")
        config = InverseGammaPriorHyperparametersConfig(alpha=alpha, beta=beta)
        inverse_gamma_prior = InverseGammaPriorHyperparameters(config=config)

        # Test values: theta must be positive
        test_values = [0.1, 0.5, 1.0, 2.0, 5.0]

        print("Comparing prior evaluations with scipy.stats.invgamma:")
        for val in test_values:
            p_dalia = inverse_gamma_prior.evaluate_prior(val)
            # scipy's invgamma uses scale parameter
            p_scipy = invgamma.pdf(val, a=alpha, scale=beta)
            print(
                f"  θ = {val:4.1f}: DALIA p = {p_dalia:.6f}, "
                f"scipy p = {p_scipy:.6f}, diff = {abs(p_dalia - p_scipy):.2e}"
            )
            if abs(p_dalia - p_scipy) > 1e-6:
                raise ValueError(
                    "Prior evaluation does not match scipy implementation."
                )

        print("Comparing log prior evaluations with scipy.stats.invgamma:")
        for val in test_values:
            log_p_dalia = inverse_gamma_prior.evaluate_log_prior(val)
            log_p_scipy = invgamma.logpdf(val, a=alpha, scale=beta)
            print(
                f"  θ = {val:4.1f}: DALIA log p = {log_p_dalia:.6f}, "
                f"scipy log p = {log_p_scipy:.6f}, diff = {abs(log_p_dalia - log_p_scipy):.2e}"
            )
            if abs(log_p_dalia - log_p_scipy) > 1e-6:
                raise ValueError(
                    "Log prior evaluation does not match scipy implementation."
                )

        # Test internal log prior via sampling
        N = 1000000
        theta_external_samples = invgamma.rvs(a=alpha, scale=beta, size=N)
        theta_internal_samples = (
            inverse_gamma_prior.rescale_hyperparameters_to_internal(
                theta_external_samples, "forward"
            )
        )

        y_min, y_max = (
            theta_internal_samples.min(),
            theta_internal_samples.max(),
        )
        counts, bins = np.histogram(
            theta_internal_samples, bins=500, range=(y_min, y_max), density=True
        )

        y_grid = np.linspace(y_min, y_max, 1000)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Filter out bins with 0 counts to prevent log(0) -> -inf errors
        valid_bins = counts > 0
        empirical_log_density = np.log(counts[valid_bins])
        filtered_bin_centers = bin_centers[valid_bins]

        # Compute theoretical log-density in internal space
        theoretical_log_density = inverse_gamma_prior.evaluate_internal_log_prior(
            y_grid
        )

        from matplotlib import pyplot as plt

        # Plot Direct Log-Space Comparison
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

        # Plot theoretical log-density function
        plt.plot(
            y_grid,
            theoretical_log_density,
            color="crimson",
            lw=2.5,
            label="Inverse Gamma Log-Density (internal space)",
        )

        plt.title(
            f"Inverse Gamma Prior Internal Log-Space Validation (α={alpha}, β={beta})",
            fontsize=14,
        )
        plt.xlabel("y (log theta)", fontsize=12)
        plt.ylabel("Log-Density", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.show()

    ######### Check forward transformation
    # Suppose I have now fitted a Gaussian in internal space and want to check that the
    # forward transformation to external space gives the correct density shape.
    # Do this by sampling from the Gaussian in internal space, transforming to external space using the gamma prior's rescaling function,
    # and comparing the empirical density of the transformed samples to the theoretical density computed using the gamma prior's evaluate_prior() method.

    # Create a inverse gamma prior configuration
    mean_values = [-1.0, 3.0, 5.0]
    sd_values = [0.5, 1.0, 2.0]

    for mean, sd in zip(mean_values, sd_values):
        N = 1000000
        internal_samples = np.random.normal(loc=mean, scale=sd, size=N)
        external_samples = inverse_gamma_prior.rescale_hyperparameters_to_internal(
            internal_samples, "backward"
        )  # Forward transformation: y = g(x)

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
            inverse_gamma_prior.rescale_hyperparameters_to_internal(
                external_grid, "forward"
            ),
            loc=mean,
            scale=sd,
        ) * np.abs(
            inverse_gamma_prior.rescale_hyperparameters_to_internal(
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
