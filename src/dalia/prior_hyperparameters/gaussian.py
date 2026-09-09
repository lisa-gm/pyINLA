# Copyright 2024-2025 DALIA authors. All rights reserved.
import numpy as np
from scipy.sparse import spmatrix
from dalia import sp, xp

from dalia import NDArray
from dalia.configs.priorhyperparameters_config import (
    GaussianPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class GaussianPriorHyperparameters(PriorHyperparameters):
    """
    Univariate Gaussian prior hyperparameters.

    This class implements prior hyperparameters following a univariate normal
    (Gaussian) distribution with specified mean and precision (inverse variance).

    Parameters
    ----------
    config : GaussianPriorHyperparametersConfig
        Configuration object containing mean and precision parameters.

    Attributes
    ----------
    mean : float
        Mean of the Gaussian distribution.
    precision : float
        Precision (inverse variance) of the Gaussian distribution.
    normalizing_constant : float
        Precomputed normalizing constant for log probability evaluation.
    """

    def __init__(
        self,
        config: GaussianPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Gaussian prior hyperparameters.

        Parameters
        ----------
        config : GaussianPriorHyperparametersConfig
            Configuration containing mean and precision parameters.

        Raises
        ------
        ValueError
            If the precision is not positive.
        """
        super().__init__(config)

        self.mean: float = config.mean
        self.precision: float = config.precision

        # Validate precision is positive
        if self.precision <= 0:
            raise ValueError(f"Precision must be positive, got {self.precision}")

        self.log_normalizing_constant = -0.5 * xp.log(2 * xp.pi) + 0.5 * xp.log(
            self.precision
        )

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Rescale hyperparameters between internal and external representations.

        Parameters
        ----------
        theta : NDArray or float
            Hyperparameter values to rescale.
        direction : str
            Direction of rescaling ('forward' or 'backward', i.e. from interpretable/user/external to internal or vice versa).

        Returns
        -------
        NDArray or float
            Rescaled hyperparameter values. In this case it is the identity, therefore unchanged.

        Notes
        -----
        For Gaussian priors, the rescaling is the identity function since the internal and external representations are the same.
        """
        return super().rescale_hyperparameters_to_internal(theta, direction)

    def evaluate_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the prior probability density.

        Computes the probability density of the univariate normal distribution at the given theta value.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the prior.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Prior probability density at theta.

        Notes
        -----
        The computation follows:
            p(θ) = C * exp(-0.5 * τ * (θ - μ)²)
        where C is the normalizing constant, τ is precision, and μ is the mean.
        """

        return xp.exp(
            self.log_normalizing_constant
            - 0.5 * self.precision * (theta - self.mean) ** 2
        )

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density.

        Computes the log probability density of the univariate normal
        distribution at the given theta value.

        Parameters
        ----------
        theta : float
            Parameter value at which to evaluate the log prior.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at theta.

        Notes
        -----
        The computation follows:
            log p(θ) = C - 0.5 * τ * (θ - μ)²
        where C is the normalizing constant, τ is precision, and μ is the mean.
        """
        return (
            self.log_normalizing_constant
            - 0.5 * self.precision * (theta - self.mean) ** 2
        )

    def evaluate_internal_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density in internal space.

        Since the Gaussian distribution is naturally defined on the unconstrained
        real line, the internal and external representations are identical. The
        log-Jacobian correction is zero (since the Jacobian of the identity
        transformation is 1, and log(1) = 0).

        Parameters
        ----------
        theta : float
            Parameter value in INTERNAL representation.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density in internal space.

        Notes
        -----
        For Gaussian priors with identity transformation:
            log p(θ_internal) = log p(θ_external) + log|dθ/dθ_internal|
                              = log p(θ_external) + 0
                              = log p(θ_external)
        """

        # Since the transformation is identity, the Jacobian correction is 0
        theta_external = self.rescale_hyperparameters_to_internal(theta, "backward")

        transformed_log_prior = self.evaluate_log_prior(
            theta_external
        ) + self.rescale_hyperparameters_to_internal(theta, "backward_log_jacobian")

        return transformed_log_prior


if __name__ == "__main__":
    """
    Test Gaussian prior hyperparameters with scipy validation.

    Validates:
    1. Prior evaluation against scipy.stats.norm
    2. Log-prior evaluation against scipy.stats.norm.logpdf
    3. Rescaling functions (identity transformation)
    4. Internal log-prior via empirical sampling
    """

    from scipy.stats import norm

    print("=" * 80)
    print("Testing Gaussian Prior Hyperparameters")
    print("=" * 80)

    # Test configurations
    mean_values = [0.0, 1.0, -2.0]
    precision_values = [0.5, 1.0, 2.0]

    for mean, precision in zip(mean_values, precision_values):
        print(f"\nTesting mean={mean}, precision={precision}")
        config = GaussianPriorHyperparametersConfig(mean=mean, precision=precision)
        gaussian_prior = GaussianPriorHyperparameters(config=config)

        # Standard deviation from precision: σ = 1/√τ
        std_dev = 1.0 / xp.sqrt(precision)

        # Test values spanning multiple standard deviations
        test_values = [
            mean - 3 * std_dev,
            mean - std_dev,
            mean,
            mean + std_dev,
            mean + 3 * std_dev,
        ]

        print("Comparing prior evaluations with scipy.stats.norm:")
        for val in test_values:
            p_dalia = gaussian_prior.evaluate_prior(val)
            p_scipy = norm.pdf(val, loc=mean, scale=std_dev)
            print(
                f"  θ = {val:7.3f}: DALIA p = {p_dalia:.6f}, "
                f"scipy p = {p_scipy:.6f}, diff = {abs(p_dalia - p_scipy):.2e}"
            )
            if abs(p_dalia - p_scipy) > 1e-6:
                raise ValueError(
                    "Prior evaluation does not match scipy implementation."
                )

        print("Comparing log prior evaluations with scipy.stats.norm:")
        for val in test_values:
            log_p_dalia = gaussian_prior.evaluate_log_prior(val)
            log_p_scipy = norm.logpdf(val, loc=mean, scale=std_dev)
            print(
                f"  θ = {val:7.3f}: DALIA log p = {log_p_dalia:.6f}, "
                f"scipy log p = {log_p_scipy:.6f}, diff = {abs(log_p_dalia - log_p_scipy):.2e}"
            )
            if abs(log_p_dalia - log_p_scipy) > 1e-6:
                raise ValueError(
                    "Log prior evaluation does not match scipy implementation."
                )

        # Test internal log prior via sampling
        # For Gaussian with identity transformation, internal and external spaces are identical
        N = 1000000
        theta_external_samples = np.random.normal(loc=mean, scale=std_dev, size=N)
        theta_internal_samples = gaussian_prior.rescale_hyperparameters_to_internal(
            theta_external_samples, "forward"
        )

        y_min, y_max = theta_internal_samples.min(), theta_internal_samples.max()
        counts, bins = np.histogram(
            theta_internal_samples, bins=500, range=(y_min, y_max), density=True
        )

        y_grid = np.linspace(y_min, y_max, 1000)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Filter out bins with 0 counts to prevent log(0) -> -inf errors
        valid_bins = counts > 0
        empirical_log_density = xp.log(counts[valid_bins])
        filtered_bin_centers = bin_centers[valid_bins]

        # Compute theoretical log-density in internal space
        theoretical_log_density = gaussian_prior.evaluate_internal_log_prior(y_grid)

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
            label="Gaussian Log-Density (internal space)",
        )

        plt.title(
            f"Gaussian Prior Internal Log-Space Validation (μ={mean}, τ={precision})",
            fontsize=14,
        )
        plt.xlabel("θ_internal", fontsize=12)
        plt.ylabel("Log-Density", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.show()

    ######### Check forward transformation
    # Sample from a Gaussian in internal space and verify forward projection to external space
    # gives the correct density shape.

    print("\n" + "=" * 80)
    print("Testing Forward Transformation (Internal → External)")
    print("=" * 80)

    mean_values_internal = [0.0, 1.0, -1.0]
    sd_values_internal = [0.5, 1.0, 0.8]

    for mean_int, sd_int in zip(mean_values_internal, sd_values_internal):
        print(f"\nTesting internal Gaussian: mean={mean_int}, sd={sd_int}")

        # Create a Gaussian prior (will be used for its rescaling and evaluation functions)
        config = GaussianPriorHyperparametersConfig(mean=0.0, precision=1.0)
        gaussian_prior = GaussianPriorHyperparameters(config=config)

        N = 1000000
        # Sample from Gaussian in internal space
        internal_samples = np.random.normal(loc=mean_int, scale=sd_int, size=N)
        # Apply forward transformation (identity for Gaussian)
        external_samples = gaussian_prior.rescale_hyperparameters_to_internal(
            internal_samples, "backward"
        )

        x_min, x_max = external_samples.min(), external_samples.max()
        counts, bins = np.histogram(
            external_samples, bins=500, range=(x_min, x_max), density=True
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2

        valid_bins = counts > 0
        empirical_density = counts[valid_bins]
        filtered_bin_centers = bin_centers[valid_bins]

        external_grid = np.linspace(x_min, x_max, 1000)

        from scipy.stats import norm

        # Since rescaling is identity, the transformed density is just the original Gaussian
        theoretical_density = norm.pdf(
            gaussian_prior.rescale_hyperparameters_to_internal(
                external_grid, "forward"
            ),
            loc=mean_int,
            scale=sd_int,
        ) * np.abs(
            gaussian_prior.rescale_hyperparameters_to_internal(
                external_grid, "forward_jacobian"
            )
        )

        from matplotlib import pyplot as plt

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
            f"Gaussian Prior Forward Transformation (Internal μ={mean_int}, σ={sd_int})",
            fontsize=14,
        )
        plt.xlabel("x (External Space)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.show()

    print("\nAll value direct value comparisons passed. Check sampling plots.")
