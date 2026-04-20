# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import xp
from dalia.configs.priorhyperparameters_config import (
    PenalizedComplexityPriorHyperparametersConfig,
)
from dalia.core.prior_hyperparameters import PriorHyperparameters


class PenalizedComplexityPriorHyperparameters(PriorHyperparameters):
    """Penalized Complexity prior hyperparameters.

    Notes
    -----
    All parameters are expected in interpretable, i.e. external scale (not log-scale).
    """

    def __init__(
        self,
        config: PenalizedComplexityPriorHyperparametersConfig,
        **kwargs,
    ) -> None:
        """Initializes the Penalized Complexity prior hyperparameters."""
        super().__init__(config)

        self.hyperparameter_type: str = kwargs.get("hyperparameter_type")

        self.alpha: float = config.alpha
        self.u: float = config.u

        self.lambda_theta: float = 0.0

        if self.hyperparameter_type == "r_s":
            spatial_dim: int = 2  # kwargs["spatial_dim", 2]

            self.lambda_theta = -xp.log(self.alpha) * pow(
                self.u,
                0.5 * spatial_dim,
            )
        elif self.hyperparameter_type == "r_t":
            self.lambda_theta = -xp.log(self.alpha) * pow(self.u, 0.5)
        elif (
            self.hyperparameter_type == "sigma_st"
            or self.hyperparameter_type == "sigma_e"
        ):
            self.lambda_theta = -xp.log(self.alpha) / self.u
        elif self.hyperparameter_type == "prec_o":
            self.lambda_theta = -xp.log(self.alpha) / self.u

        # print("lambda_theta: ", self.lambda_theta)

    # def __init__(
    #     self,
    #     config: PenalizedComplexityPriorHyperparametersConfig,
    #     **kwargs,
    # ) -> None:
    #     """Initializes the Penalized Complexity prior hyperparameters."""
    #     super().__init__(config)

    #     self.hyperparameter_type: str = kwargs.get("hyperparameter_type")

    #     self.alpha: float = config.alpha
    #     self.u: float = config.u

    #     self.lambda_theta: float = 0.0

    #     if self.hyperparameter_type == "r_s":
    #         spatial_dim: int = 2  # kwargs["spatial_dim", 2]

    #         self.lambda_theta = -xp.log(self.alpha) * pow(
    #             self.u,
    #             0.5 * spatial_dim,
    #         )
    #     elif self.hyperparameter_type == "r_t":
    #         self.lambda_theta = -xp.log(self.alpha) * pow(self.u, 0.5)
    #     elif (
    #         self.hyperparameter_type == "sigma_st"
    #         or self.hyperparameter_type == "sigma_e"
    #     ):
    #         self.lambda_theta = -xp.log(self.alpha) / self.u
    #     elif self.hyperparameter_type == "prec_o":
    #         self.lambda_theta = -xp.log(self.alpha) / self.u

    #     # print("lambda_theta: ", self.lambda_theta)

    # def rescale_hyperparameters_to_internal(self, theta, direction):
    #     return super().rescale_hyperparameters_to_internal(theta, direction)

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
            If direction is not "forward" or "backward".
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

    def evaluate_log_prior(self, theta_external: float, **kwargs) -> float:
        """Evaluate the prior hyperparameters."""
        log_prior: float = 0.0

        print("theta_external: ", theta_external)
        ## should be converted to internal scale
        theta = self.rescale_hyperparameters_to_internal(theta_external, "backward")

        if self.hyperparameter_type == "r_s":
            spatial_dim: int = 2  # kwargs["spatial_dim", 2]

            if spatial_dim == 2:
                log_prior = (
                    xp.log(self.lambda_theta)
                    - self.lambda_theta * xp.exp(-theta)
                    - theta
                )
            else:
                raise ValueError("Not implemented for other than 2D spatial domains")
            # print("log prior r s: ", log_prior)
        elif self.hyperparameter_type == "r_t":
            log_prior = (
                xp.log(self.lambda_theta)
                - self.lambda_theta * xp.exp(-0.5 * theta)
                + xp.log(0.5)
                - 0.5 * theta
            )
            # print("log prior r t: ", log_prior)
        elif (
            self.hyperparameter_type == "sigma_st"
            or self.hyperparameter_type == "sigma_e"
        ):
            log_prior = (
                xp.log(self.lambda_theta) - self.lambda_theta * xp.exp(theta) + theta
            )
            # print("log prior sigma e: ", log_prior)
        elif self.hyperparameter_type == "prec_o":
            # log_prior = (
            #     xp.log(self.lambda_theta) - self.lambda_theta * xp.exp(theta) + theta
            # )

            # according to inla docs
            log_prior = (0.5 * self.lambda_theta) * xp.exp(
                -self.lambda_theta * xp.exp(-0.5 * theta) - 0.5 * theta
            )
            print("log prior prec o: ", log_prior)

        self.lambda_theta = -xp.log(self.alpha) / self.u

        # according to inla docs
        log_prior = (0.5 * self.lambda_theta) * xp.exp(
            -self.lambda_theta * xp.exp(-0.5 * theta) - 0.5 * theta
        )
        print("log prior prec o: ", log_prior)

        # add correction for change of variables
        print("log prior before jacobian correction: ", log_prior)
        log_prior += xp.log(
            xp.abs(self.rescale_hyperparameters_to_internal(theta, "forward_jacobian"))
        )
        print("log prior after jacobian correction: ", log_prior)

        log_prior1 = (
            xp.log(self.lambda_theta)
            - self.lambda_theta * xp.exp(theta_external)
            + theta_external
        )
        print("log prior in external scale: ", log_prior1)

        return log_prior


if __name__ == "__main__":

    alpha = 0.02
    u = 6

    tau = 4.0

    lambda_theta = -xp.log(alpha) / u

    log_tau = xp.log(tau)
    print("tau: ", tau, ", log(tau): ", log_tau, ", lambda_theta: ", lambda_theta)

    pc_prior = (
        0.5
        * lambda_theta
        * xp.power(tau, -1.5)
        * xp.exp(-lambda_theta * xp.power(tau, -0.5))
    )
    print(f"PC prior density at tau={tau}: {pc_prior}, log form: {xp.log(pc_prior)}")

    pc_prior_log_theta = (
        0.5
        * lambda_theta
        * xp.exp(-lambda_theta * xp.exp(-0.5 * log_tau) - 0.5 * log_tau)
        * xp.abs(1 / tau)
    )
    print(f"Log PC prior density at tau={tau}: {pc_prior_log_theta}")

    log_pc_prior_log_theta = (
        xp.log(0.5)
        + xp.log(lambda_theta)
        - lambda_theta * xp.exp(-0.5 * log_tau)
        - 0.5 * log_tau
        - xp.log(xp.abs(tau))
    )

    print(f"Log PC prior density at tau={tau} (log form): {log_pc_prior_log_theta}")
