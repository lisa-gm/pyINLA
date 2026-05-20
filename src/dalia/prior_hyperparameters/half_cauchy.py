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

    We use the internal parameterization:

    theta = log(1 / sigma^2) = -2 * log(sigma)

    and evaluate the prior in internal space through change of variables:

    log p(theta) = log p(sigma(theta)) + log(|d sigma / d theta|)

    Parameters
    ----------
    config : HalfCauchyPriorHyperparametersConfig
        Configuration object containing the scale parameter.

    Attributes
    ----------
    scale : float. scale > 0. Default is 25.0.
        Scale parameter of the Half-Cauchy distribution.
    normalizing_constant : float
        Precomputed constant term for log probability evaluation in internal theta-space.
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

        The Half-Cauchy prior is defined on sigma > 0 and the internal variable is
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
        elif (
            direction == "forward_jacobian"
        ):  ## for better naming should be log jacobian
            theta_scaled = -2.0 / theta
        elif direction == "backward_jacobian":
            theta_scaled = -xp.log(2.0) - 0.5 / theta
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return theta_scaled

    def evaluate_log_prior(self, theta: float, **kwargs) -> float:
        """
        Evaluate the log prior probability density.

        Computes the log probability density in external space where
        theta = sigma.

        Parameters
        ----------
        theta : float
            External value theta = sigma
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Log prior probability density at internal theta.
        """

        theta_internal = self.rescale_hyperparameters_to_internal(theta, "forward")

        print(
            f"Evaluating Half-Cauchy log prior at sigma = {theta:.3f} with jacobian = {self.rescale_hyperparameters_to_internal(theta_internal, 'backward_jacobian'):.6f}"
        )

        log_prior = (
            self.normalizing_constant
            - xp.log(1 + (theta / self.scale) ** 2)
            + self.rescale_hyperparameters_to_internal(
                theta_internal, direction="backward_jacobian"
            )
        )

        return log_prior


if __name__ == "__main__":
    """
    Basic checks for Half-Cauchy prior implementation in internal theta-space.
    """

    from scipy.stats import halfcauchy

    scale_values = [0.5, 1.0, 2.0]

    for scale in scale_values:
        print(f"\nTesting scale={scale}")
        config = HalfCauchyPriorHyperparametersConfig(scale=scale)
        half_cauchy_prior = HalfCauchyPriorHyperparameters(config=config)

        test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        print(
            "Comparing log prior evaluations in internal theta-space with scipy.stats.halfcauchy + Jacobian:"
        )
        for sigma in test_values:
            theta_internal = half_cauchy_prior.rescale_hyperparameters_to_internal(
                sigma, "forward"
            )
            log_jac = xp.log(
                xp.abs(-0.5 * np.exp(-theta_internal / 2.0))
            )  # np.log(np.abs(-0.5 * sigma))  # log(|d sigma / d theta|)
            print(
                " Testing sigma = {:.1f} (theta_internal = {:.3f}), log_jac = {:.6f}:".format(
                    sigma, theta_internal, log_jac
                )
            )

            logp_dalia = half_cauchy_prior.evaluate_log_prior(sigma)
            logp_scipy = halfcauchy.logpdf(sigma, loc=0.0, scale=scale) + log_jac
            print(
                f"  sigma = {sigma:4.1f}: DALIA logp = {logp_dalia:.6f}, "
                f"scipy logp = {logp_scipy:.6f}, diff = {abs(logp_dalia - logp_scipy):.6e}"
            )
            if abs(logp_dalia - logp_scipy) > 1e-6:
                raise ValueError(
                    "Log prior evaluation does not match scipy implementation."
                )

    print("\nAll Half-Cauchy checks passed!")
