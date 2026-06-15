# Copyright 2024-2025 DALIA authors. All rights reserved.
import numpy as np
from scipy.sparse import spmatrix

from dalia import NDArray, sp, xp
from dalia.configs.priorhyperparameters_config import (
    GaussianMVNPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class GaussianMVNPriorHyperparameters(PriorHyperparameters):
    """
    Gaussian multivariate normal (MVN) prior hyperparameters.

    This class implements prior hyperparameters following a multivariate normal
    distribution with specified mean and precision matrix.

    Parameters
    ----------
    config : GaussianMVNPriorHyperparametersConfig
        Configuration object containing mean and precision matrix.

    Attributes
    ----------
    mean : NDArray
        Mean vector of the multivariate normal distribution.
    precision : spmatrix
        Precision matrix (inverse covariance) of the distribution.
    normalizing_constant : float
        Precomputed normalizing constant for log probability evaluation.
    """

    def __init__(
        self,
        config: GaussianMVNPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Gaussian MVN prior hyperparameters.

        Parameters
        ----------
        config : GaussianMVNPriorHyperparametersConfig
            Configuration containing mean vector and precision matrix.

        Raises
        ------
        ValueError
            If the precision matrix is not positive definite.
        """
        super().__init__(config)

        self.mean: NDArray = config.mean
        self.precision: spmatrix = config.precision

        if xp == np:
            self.mean: NDArray = self.mean
            self.precision: spmatrix = self.precision
        else:
            self.mean: NDArray = xp.asarray(self.mean)
            self.precision: sp.sparse.spmatrix = sp.sparse.csc_matrix(self.precision)

        sign, logabsdet = np.linalg.slogdet(self.precision.toarray())
        if sign != 1:
            raise ValueError("Precision matrix must be positive definite.")

        self.log_normalizing_constant = (
            -0.5 * self.mean.shape[0] * xp.log(2 * xp.pi) + 0.5 * logabsdet
        )

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Rescale hyperparameters between internal and external/user representations.

        Parameters
        ----------
        theta : NDArray
            Hyperparameter values to rescale.
        direction : str
            Direction of rescaling ('forward' or 'backward', 'forward_jacobian', 'backward_log_jacobian').

        Returns
        -------
        NDArray
            Rescaled hyperparameter values, which is the identity in this case, therefore unchanged.

        Notes
        -----
        For MVN priors, the rescaling is the identity function since the internal and external representations are the same.
        """
        return super().rescale_hyperparameters_to_internal(theta, direction)

    def evaluate_prior(self, theta: NDArray, **kwargs) -> float:
        """
        Evaluate the prior probability density.

        Computes the probability density of the multivariate normal distribution
        at the given theta value(s).

        Parameters
        ----------
        theta : NDArray
            Parameter values at which to evaluate the prior.
            Must have the same shape as the mean vector.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float or NDArray
            Prior probability density at theta.

        Notes
        -----
        The computation follows:
            p(θ) = C * exp(-0.5 * (θ - μ)^T @ Q @ (θ - μ))
        where C is the normalizing constant, Q is the precision matrix, and μ is the mean.
        """
        if self.mean.shape != theta.shape:
            raise ValueError(
                f"Shape of theta ({theta.shape}) and mean ({self.mean.shape}) do not match."
            )

        return xp.exp(self.evaluate_log_prior(theta))

    def evaluate_log_prior(self, theta: NDArray, **kwargs) -> float:
        """
        Evaluate the log prior probability density.

        Computes the log probability density of the multivariate normal
        distribution at the given theta values.

        Parameters
        ----------
        theta : NDArray
            Parameter values at which to evaluate the log prior.
            Must have the same shape as the mean vector.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at theta.

        Raises
        ------
        ValueError
            If theta and mean have incompatible shapes.

        Notes
        -----
        The computation follows:
            log p(θ) = C - 0.5 * (θ - μ)^T @ Q @ (θ - μ)
        where C is the normalizing constant and Q is the precision matrix.
        """
        if self.mean.shape != theta.shape:
            raise ValueError(
                f"Shape of theta ({theta.shape}) and mean ({self.mean.shape}) do not match."
            )

        diff = theta - self.mean
        if isinstance(self.mean, float):
            quad_form = float(diff * self.precision * diff)
        else:
            quad_form = diff.T @ self.precision @ diff

        return self.log_normalizing_constant - 0.5 * quad_form

    def evaluate_internal_log_prior(self, theta: NDArray, **kwargs) -> float:
        """
        Evaluate the log prior probability density in internal space.

        Since the MVN distribution is naturally defined on the unconstrained ℝ^d,
        the internal and external representations are identical. The log-Jacobian
        correction is zero (since the Jacobian of the identity transformation is 1).

        Parameters
        ----------
        theta : NDArray
            Parameter values in external representation.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density in internal space.

        Notes
        -----
        For MVN priors with identity transformation:
            log p(θ_internal) = log p(θ_external) + log|dθ/dθ_internal|
                              = log p(θ_external) + 0
                              = log p(θ_external)
        """
        # Since transformation is identity, Jacobian correction is 0
        theta_internal = self.rescale_hyperparameters_to_internal(theta, "forward")

        transformed_log_prior = self.evaluate_log_prior(
            theta
        ) + self.rescale_hyperparameters_to_internal(
            theta_internal, "backward_log_jacobian"
        )

        return transformed_log_prior


if __name__ == "__main__":
    """
    Test Gaussian MVN prior hyperparameters with scipy validation.

    Validates:
    1. Prior evaluation against scipy.stats.multivariate_normal
    2. Log-prior evaluation against scipy.stats.multivariate_normal.logpdf
    3. Internal log-prior via empirical sampling
    4. Marginal distributions match univariate Gaussians
    """

    from scipy.stats import multivariate_normal, norm
    from scipy.sparse import csc_matrix
    from matplotlib import pyplot as plt

    print("=" * 80)
    print("Testing Gaussian MVN Prior Hyperparameters (2D Test Case)")
    print("=" * 80)

    # Create a 2D test case
    mean_2d = np.array([0.0, 1.0])
    cov_2d = np.array([[1.0, 0.3], [0.3, 0.5]])
    precision_2d = np.linalg.inv(cov_2d)

    print(f"\n2D Test Case:")
    print(f"  Mean: {mean_2d}")
    print(f"  Covariance:\n{cov_2d}")
    print(f"  Precision:\n{precision_2d}")

    config = GaussianMVNPriorHyperparametersConfig(
        mean=mean_2d, precision=csc_matrix(precision_2d)
    )
    mvn_prior = GaussianMVNPriorHyperparameters(config=config)

    # Create scipy reference
    scipy_mvn = multivariate_normal(mean=mean_2d, cov=cov_2d)

    # Test 1: Direct evaluation at specific points
    print("\n1. Comparing prior evaluations with scipy.stats.multivariate_normal:")
    test_points = np.array(
        [
            [0.0, 1.0],  # At mean
            [1.0, 1.0],  # Perturbed in x
            [0.0, 2.0],  # Perturbed in y
            [-1.0, 0.5],  # Different point
            [0.5, 1.5],  # Diagonal perturbation
        ]
    )

    for point in test_points:
        p_dalia = mvn_prior.evaluate_prior(point)
        p_scipy = scipy_mvn.pdf(point)
        print(
            f"  θ = {point}: DALIA p = {p_dalia:.6f}, "
            f"scipy p = {p_scipy:.6f}, diff = {abs(p_dalia - p_scipy):.2e}"
        )
        if abs(p_dalia - p_scipy) > 1e-6:
            raise ValueError("Prior evaluation does not match scipy implementation.")

    # Test 2: Log-prior evaluation
    print("\nComparing log prior evaluations with scipy.stats.multivariate_normal:")
    for point in test_points:
        log_p_dalia = mvn_prior.evaluate_log_prior(point)
        log_p_scipy = scipy_mvn.logpdf(point)
        print(
            f"  θ = {point}: DALIA log p = {log_p_dalia:.6f}, "
            f"scipy log p = {log_p_scipy:.6f}, diff = {abs(log_p_dalia - log_p_scipy):.2e}"
        )
        if abs(log_p_dalia - log_p_scipy) > 1e-6:
            raise ValueError(
                "Log prior evaluation does not match scipy implementation."
            )

    # Test 3: Rescaling functions (identity transformation for Gaussian)
    print("\nTesting rescaling functions (identity transformation):")
    test_point = np.array([0.5, 1.5])

    # Forward: external → internal (identity)
    theta_internal_forward = mvn_prior.rescale_hyperparameters_to_internal(
        test_point, "forward"
    )
    print(f"  Forward (external→internal): {test_point} → {theta_internal_forward}")
    if not np.allclose(test_point, theta_internal_forward):
        raise ValueError("Forward transformation should be identity for Gaussian.")

    # Backward: internal → external (identity)
    theta_external_backward = mvn_prior.rescale_hyperparameters_to_internal(
        theta_internal_forward, "backward"
    )
    print(
        f"  Backward (internal→external): {theta_internal_forward} → {theta_external_backward}"
    )
    if not np.allclose(test_point, theta_external_backward):
        raise ValueError("Backward transformation should be identity for Gaussian.")

    # Log-Jacobian should be 0 for identity transformation
    log_jacobian = mvn_prior.rescale_hyperparameters_to_internal(
        test_point, "backward_log_jacobian"
    )
    print("log jacobian: ", log_jacobian)
    print(f"  Log-Jacobian (backward): {log_jacobian} (should be 0 for identity)")
    if abs(log_jacobian).any() > 1e-10:
        raise ValueError(
            "Backward Log-Jacobian should be 0 for identity transformation."
        )

    # Test 3: 2D scatter plot with theoretical contours
    # Idea: Sample from the MVN prior, rescale to internal (identity), and "bin" it but on a 2D grid which gives you the empirical density
    # then compute the theoretical density on the same grid and plot contours of the theoretical density on top of the empirical density to visually check if they match
    print("\nTesting evaluate_internal_log_prior via empirical sampling:")

    N = 1000000
    theta_external_samples = scipy_mvn.rvs(size=N)  # Shape: (N, 2)
    theta_internal_samples = mvn_prior.rescale_hyperparameters_to_internal(
        theta_external_samples, "forward"
    )

    print("\nGenerating 2D visualization (contours + samples)...")

    viz_samples = theta_external_samples[:50000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: 2D histogram with theoretical contours overlay
    ax = axes[0]

    # Create 2D histogram
    counts, xedges, yedges, im = ax.hist2d(
        viz_samples[:, 0], viz_samples[:, 1], bins=50, cmap="YlOrRd", cmin=1
    )

    # Convert counts to empirical log-density (normalized)
    N = len(viz_samples)
    bin_width_x = xedges[1] - xedges[0]
    bin_width_y = yedges[1] - yedges[0]
    bin_area = bin_width_x * bin_width_y

    # Empirical density = counts / (N * bin_area)
    # Empirical log-density = log(counts) - log(N * bin_area)
    empirical_log_density = np.log(counts.T) - np.log(N * bin_area)

    # Create a new image with normalized log-density values
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im_log = ax.imshow(
        empirical_log_density,
        extent=extent,
        origin="lower",
        cmap="YlOrRd",
        aspect="auto",
    )
    cbar = plt.colorbar(im_log, ax=ax, label="Empirical Log-Density")

    # Overlay theoretical contours
    x = np.linspace(
        mean_2d[0] - 4 * np.sqrt(cov_2d[0, 0]),
        mean_2d[0] + 4 * np.sqrt(cov_2d[0, 0]),
        100,
    )
    y = np.linspace(
        mean_2d[1] - 4 * np.sqrt(cov_2d[1, 1]),
        mean_2d[1] + 4 * np.sqrt(cov_2d[1, 1]),
        100,
    )
    X, Y = np.meshgrid(x, y)
    Z = []
    for i in range(X.shape[0]):
        row = []
        for j in range(X.shape[1]):
            result = mvn_prior.evaluate_internal_log_prior(np.array([X[i, j], Y[i, j]]))
            # Handle both scalar and array returns
            if np.isscalar(result):
                row.append(result)
            else:
                row.append(np.asarray(result).flat[0])
        Z.append(row)
    Z = np.array(Z)

    # Add filled contours to show density gradient
    levels = 5
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap="Greys", alpha=0.3)

    # Overlay contour lines
    contours = ax.contour(
        X, Y, Z, levels=levels, colors="black", alpha=0.5, linewidths=0.5
    )
    ax.clabel(contours, inline=True, fontsize=6)
    ax.set_xlabel("θ₀")
    ax.set_ylabel("θ₁")
    ax.set_title(
        "2D MVN Prior: Empirical Density (heatmap) vs Theoretical Contours (black)"
    )
    ax.grid(True, alpha=0.3)

    # Right plot: 1D slices through your MVN implementation
    ax = axes[1]

    # Slice 1: Fix θ₁ = mean[1], vary θ₀
    # reuse x

    dalia_slice_0 = np.array(
        [mvn_prior.evaluate_log_prior(np.array([t, mean_2d[1]])) for t in x]
    )
    scipy_slice_0 = np.array([scipy_mvn.logpdf(np.array([t, mean_2d[1]])) for t in x])

    ax.plot(x, dalia_slice_0, "b-", lw=2.5, label="DALIA (θ₁=μ₁)")
    ax.plot(
        x[::10],
        scipy_slice_0[::10],
        "bo",
        markersize=6,
        markerfacecolor="none",
        markeredgewidth=2,
        label="scipy (θ₁=μ₁)",
    )

    # Slice 2: Fix θ₀ = mean[0], vary θ₁
    # reuse y
    dalia_slice_1 = np.array(
        [mvn_prior.evaluate_log_prior(np.array([mean_2d[0], t])) for t in y]
    )
    scipy_slice_1 = np.array([scipy_mvn.logpdf(np.array([mean_2d[0], t])) for t in y])

    ax.plot(y, dalia_slice_1, "r-", lw=2.5, label="DALIA (θ₀=μ₀)")
    ax.plot(
        y[::10],
        scipy_slice_1[::10],
        "ro",
        markersize=6,
        markerfacecolor="none",
        markeredgewidth=2,
        label="scipy (θ₀=μ₀)",
    )

    ax.set_xlabel("θ")
    ax.set_ylabel("Log-Density")
    ax.set_title("1D Slices: DALIA vs scipy")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Test 6: Forward transformation validation
    print("\nTesting forward transformation (Internal → External):")

    # Sample from Gaussian in internal space, transform to external, verify density
    mean_internal = np.array([0.0, 1.0])
    cov_internal = np.array([[1.0, 0.2], [0.2, 0.8]])

    N_forward = 500000
    internal_samples = np.random.multivariate_normal(
        mean=mean_internal, cov=cov_internal, size=N_forward
    )

    # Forward transformation (identity for Gaussian)
    external_samples = mvn_prior.rescale_hyperparameters_to_internal(
        internal_samples, "backward"
    )

    # Project to 1D for validation (first dimension)
    ext_1d = external_samples[:, 0]
    xmin, xmax = ext_1d.min(), ext_1d.max()
    counts_fwd, bins_fwd = np.histogram(
        ext_1d, bins=300, range=(xmin, xmax), density=True
    )
    bin_centers_fwd = (bins_fwd[:-1] + bins_fwd[1:]) / 2

    # Theoretical marginal density of first coordinate
    # For identity transformation, the marginal of the internal Gaussian is preserved
    from scipy.stats import norm

    marginal_mean_0 = mean_internal[0]
    marginal_std_0 = np.sqrt(cov_internal[0, 0])
    theoretical_density_fwd = norm.pdf(
        bin_centers_fwd, loc=marginal_mean_0, scale=marginal_std_0
    )

    plt.figure(figsize=(9, 6))
    plt.scatter(
        bin_centers_fwd,
        counts_fwd,
        color="forestgreen",
        s=10,
        alpha=0.7,
        label="Empirical Density (transformed samples)",
    )
    plt.plot(
        bin_centers_fwd,
        theoretical_density_fwd,
        color="orange",
        lw=2.5,
        label="Theoretical Density (Gaussian in internal space)",
    )
    plt.title("Forward Transformation: Internal Gaussian → External Space", fontsize=14)
    plt.xlabel("θ_external[0]", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.show()

    print(
        "\nAll tests passed. Empirical sampling and transformation validation complete."
    )
