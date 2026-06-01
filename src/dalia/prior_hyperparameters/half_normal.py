# Copyright 2024-2026 DALIA authors. All rights reserved.
from dalia import NDArray
from scipy.sparse import spmatrix
from dalia import xp

import numpy as np

from dalia.configs.priorhyperparameters_config import (
    HalfNormalPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class HalfNormalPriorHyperparameters(PriorHyperparameters):
    """Half-Normal prior hyperparameters.

    p(sigma) = sqrt(2 * precision / pi) * exp(-0.5 * precision * sigma^2), for sigma > 0

    and in log scale:
    log p(sigma) = 0.5 * log(2) + 0.5 * log(precision) - 0.5 * log(pi)
                   - 0.5 * precision * sigma^2

    where sigma is a positive scale parameter, typically the standard deviation.

    We use the internal parameterization:

    theta = log(1 / sigma^2) = -2 * log(sigma)

    and evaluate the prior in internal space through change of variables:

    log p(theta) = log p(sigma(theta)) + log(|d sigma / d theta|)

    Parameters
    ----------
    config : HalfNormalPriorHyperparametersConfig
        Configuration object containing the precision parameter.

    Attributes
    ----------
    precision : float. precision > 0. Default is 0.001.
        Precision parameter of the Half-Normal distribution.
    normalizing_constant : float
        Precomputed constant term for log probability evaluation.
    """

    def __init__(
        self,
        config: HalfNormalPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Half-Normal prior hyperparameters.

        Parameters
        ----------
        config : HalfNormalPriorHyperparametersConfig
            Configuration containing the precision parameter.

        Raises
        ------
        ValueError
            If precision is not positive.
        """
        super().__init__(config)

        self.precision: float = config.precision

        if self.precision <= 0:
            raise ValueError(f"Precision must be positive, got {self.precision}")

        self.normalizing_constant: float = (
            0.5 * xp.log(2.0) + 0.5 * xp.log(self.precision) - 0.5 * xp.log(xp.pi)
        )

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Transform between external and internal parameter representations.

        The Half-Normal prior is defined on sigma > 0 (standard deviation) and the internal variable is
        chosen as log precision:

        theta = log(1 / sigma^2) = -2 * log(sigma),
        sigma = exp(-0.5 * theta).

        Parameters
        ----------
        theta : float or NDArray
            Parameter value(s) to transform.
        direction : str
            Transformation direction:
            - "forward": from external sigma to internal theta
            - "backward": from internal theta to external sigma

        Returns
        -------
        float or NDArray
            Transformed parameter value(s).

        Raises
        ------
        ValueError
            If direction is not recognized.
        """
        if direction == "forward":
            theta_scaled = -2.0 * xp.log(theta)
        elif direction == "backward":
            theta_scaled = xp.exp(-0.5 * theta)
        elif direction == "forward_jacobian":
            theta_scaled = -2.0 / theta
        elif direction == "backward_log_jacobian":
            # log(|d sigma / d theta|) = log(0.5 * sigma) = -log(2) - 0.5 * theta
            theta_scaled = -xp.log(2.0) - 0.5 * theta
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density.

        Computes the log probability density in external space where
        theta = sigma, with Jacobian correction from internal mapping.

        Parameters
        ----------
        theta : float
            External value theta = sigma.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at internal theta.
        """

        theta_internal = self.rescale_hyperparameters_to_internal(theta, "forward")

        log_prior = (
            self.normalizing_constant
            - 0.5 * self.precision * theta * theta
            + self.rescale_hyperparameters_to_internal(
                theta_internal, direction="backward_log_jacobian"
            )
        )

        return log_prior


if __name__ == "__main__":
    """
    Basic checks for Half-Normal prior implementation in internal theta-space.
    """

    import matplotlib.pyplot as plt
    from scipy.stats import halfnorm

    precision_values = [0.001, 0.1, 1.0]

    for precision in precision_values:
        print(f"\nTesting precision={precision}")
        config = HalfNormalPriorHyperparametersConfig(precision=precision)
        half_normal_prior = HalfNormalPriorHyperparameters(config=config)
        scale = 1.0 / np.sqrt(precision)

        test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        print(
            "Comparing log prior evaluations in internal theta-space with scipy.stats.halfnorm + Jacobian:"
        )
        for sigma in test_values:
            theta_internal = half_normal_prior.rescale_hyperparameters_to_internal(
                sigma, "forward"
            )
            log_jac = -xp.log(2.0) + xp.log(sigma)

            logp_dalia = half_normal_prior.evaluate_log_prior(sigma)
            logp_scipy = halfnorm.logpdf(sigma, loc=0.0, scale=scale) + log_jac
            print(
                f"  sigma = {sigma:4.1f}: DALIA logp = {logp_dalia:.6f}, "
                f"scipy logp = {logp_scipy:.6f}, diff = {abs(logp_dalia - logp_scipy):.6e}"
            )
            if abs(logp_dalia - logp_scipy) > 1e-6:
                raise ValueError(
                    "Log prior evaluation does not match scipy implementation."
                )

    print("\nAll Half-Normal checks passed!")

    # Plot Half-Normal prior in external (sigma) and internal (theta) x-scales.
    half_normal_plot = HalfNormalPriorHyperparameters(
        config=HalfNormalPriorHyperparametersConfig(precision=0.001)
    )

    sigma_grid = np.logspace(-5, 2, 500)
    theta_grid = half_normal_plot.rescale_hyperparameters_to_internal(
        sigma_grid, "forward"
    )
    order = np.argsort(theta_grid)
    theta_sorted = theta_grid[order]

    # External-scale density p(sigma).
    p_sigma_half_normal = np.sqrt(2.0 * half_normal_plot.precision / np.pi) * np.exp(
        -0.5 * half_normal_plot.precision * sigma_grid * sigma_grid
    )

    # Internal-scale density p(theta) via change of variables.
    log_jac = -np.log(2.0) + np.log(sigma_grid)
    p_theta_half_normal = np.exp(
        half_normal_plot.normalizing_constant
        - 0.5 * half_normal_plot.precision * sigma_grid * sigma_grid
        + log_jac
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(sigma_grid, p_sigma_half_normal, label="Half-Normal")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("External scale (sigma)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Half-Normal Density in External Scale")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(theta_sorted, p_theta_half_normal[order], label="Half-Normal")
    axes[1].set_xlabel("Internal scale (theta = log(1/sigma^2))")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Half-Normal Density in Internal Scale")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Half-Normal (precision=0.001)", fontsize=12)
    fig.tight_layout()
    plt.show()
