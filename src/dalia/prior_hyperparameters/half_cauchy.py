# Copyright 2024-2026 DALIA authors. All rights reserved.
from dalia import NDArray
from scipy.sparse import spmatrix
from dalia import xp

import numpy as np

from dalia.configs.priorhyperparameters_config import (
    HalfCauchyPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class HalfCauchyPriorHyperparameters(PriorHyperparameters):
    """Half-Cauchy prior hyperparameters.

    p(sigma) = 2 / (pi * scale * (1 + (sigma / scale)^2)), for sigma > 0

    and in log scale:
    log p(sigma) = log(2) - log(pi) - log(scale) - log(1 + (sigma / scale)^2)

    where sigma is a positive scale parameter, typically the standard deviation.
    The model is typically defined for precision p = 1/sigma^2.

    Parameters
    ----------
    config : HalfCauchyPriorHyperparametersConfig
        Configuration object containing the scale parameter.

    Attributes
    ----------
    scale : float. scale > 0. Default is 25.0.
        Scale parameter of the Half-Cauchy distribution.
    normalizing_constant : float
        Precomputed constant term for log probability evaluation.
    """

    def __init__(
        self,
        config: HalfCauchyPriorHyperparametersConfig,
    ) -> None:
        """
        Initialize the Half-Cauchy prior hyperparameters.

        Parameters
        ----------
        config : HalfCauchyPriorHyperparametersConfig
            Configuration containing the scale parameter.

        Raises
        ------
        ValueError
            If scale is not positive.
        """
        super().__init__(config)

        self.scale: float = config.scale

        if self.scale <= 0:
            raise ValueError(f"Scale must be positive, got {self.scale}")

        self.normalizing_constant: float = (
            xp.log(2.0) - xp.log(np.pi) - xp.log(self.scale)
        )

    def rescale_hyperparameters_to_internal(self, theta, direction):
        """
        Transform between external and internal parameter representations.

        The Half-Cauchy distribution is defined for positive values, but optimization
        works in unconstrained space. This method transforms
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
            If direction is not recognized.
        """
        if direction == "forward":
            theta_scaled = xp.log(theta)
        elif direction == "backward":
            theta_scaled = xp.exp(theta)
        elif direction == "forward_jacobian":
            theta_scaled = 1 / theta  # d(log(theta))/d(theta) = 1/theta
        elif direction == "backward_jacobian":
            theta_scaled = theta  # d(exp(theta))/d(theta) = exp(theta) = theta
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density.

        Computes the log probability density of the Half-Cauchy distribution
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
        """

        z = theta / self.scale
        log_prior = self.normalizing_constant - xp.log(1 + z * z)

        return log_prior


if __name__ == "__main__":
    """
    Basic checks for Half-Cauchy prior implementation.
    """

    from scipy.stats import halfcauchy

    scale_values = [0.5, 1.0, 2.0]

    for scale in scale_values:
        print(f"\nTesting scale={scale}")
        config = HalfCauchyPriorHyperparametersConfig(scale=scale)
        half_cauchy_prior = HalfCauchyPriorHyperparameters(config=config)

        test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        print("Comparing log prior evaluations with scipy.stats.halfcauchy:")
        for val in test_values:
            logp_dalia = half_cauchy_prior.evaluate_log_prior(val)
            logp_scipy = halfcauchy.logpdf(val, loc=0.0, scale=scale)
            print(
                f"  theta = {val:4.1f}: DALIA logp = {logp_dalia:.6f}, "
                f"scipy logp = {logp_scipy:.6f}, diff = {abs(logp_dalia - logp_scipy):.2e}"
            )
            if abs(logp_dalia - logp_scipy) > 1e-6:
                raise ValueError(
                    "Log prior evaluation does not match scipy implementation."
                )

    print("\nAll Half-Cauchy checks passed!")
