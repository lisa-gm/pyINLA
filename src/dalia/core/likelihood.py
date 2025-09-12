# Copyright 2024-2025 DALIA authors. All rights reserved.

from abc import ABC, abstractmethod

from dalia import ArrayLike, NDArray
from dalia.configs.likelihood_config import LikelihoodConfig


class Likelihood(ABC):
    """Abstract core class for likelihood."""

    def __init__(
        self,
        n_observations: int,
        config: LikelihoodConfig,
    ) -> None:
        """Initializes the likelihood."""

        self.config = config
        self.n_observations = n_observations

    def gradient_likelihood(self, eta, y, h=1e-4, **kwargs):
        if self.config.method == "exact":
            return self.evaluate_gradient_likelihood(eta, y, **kwargs)
        elif self.config.method == "finite_difference":
            grad = self.finite_difference_gradient_likelihood(eta, y, h, **kwargs)
            return grad
        else:
            raise NotImplementedError(f"Method {self.config.method} not implemented.")
        # ref = self.evaluate_gradient_likelihood(eta, y, **kwargs)
        # grad = self.finite_difference_gradient_likelihood(eta, y, h, **kwargs)
        # assert xp.allclose(ref, grad), f"Gradient mismatch: {ref} vs {grad}"
        # return grad

    def hessian_likelihood(self, h: float = 1e-2, **kwargs):
        if self.config.method == "exact":
            return self.evaluate_hessian_likelihood(**kwargs)
        elif self.config.method == "finite_difference":
            kwargs = kwargs or {}
            kwargs["h"] = h
            hess = self.finite_difference_hessian_likelihood(**kwargs)
            return hess
        else:
            raise NotImplementedError(f"Method {self.config.method} not implemented.")
        # ref = self.evaluate_hessian_likelihood(**kwargs)
        # ref_diag = ref.diagonal()
        # kwargs = kwargs or {}
        # kwargs["h"] = 1e-3
        # hess = self.finite_difference_hessian_likelihood(**kwargs)
        # rel_error = xp.linalg.norm(ref_diag - hess) / xp.linalg.norm(ref_diag)
        # if not xp.allclose(ref_diag, hess):
        #     print(f"Hessian mismatch: {rel_error}")
        # return hess

    @abstractmethod
    def evaluate_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        """Evaluate the likelihood.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        **kwargs : optional
            Hyperparameters for likelihood.


        Returns
        -------
        likelihood : float
            Likelihood.
        """
        pass

    def evaluate_sum_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> float:
        """Evaluate the sum of the likelihood over all observations.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        kwargs :
            theta : float
                Specific parameter for the likelihood calculation.

        Returns
        -------
        sum_likelihood : float
            Sum of the likelihood over all observations.
        """
        likelihood = self.evaluate_likelihood(eta, y, **kwargs)
        sum_likelihood = float(likelihood.sum())
        return sum_likelihood

    def evaluate_gradient_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> NDArray:
        """Evaluate the gradient of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        **kwargs : optional
            Hyperparameters for likelihood.

        Returns
        -------
        gradient_likelihood : NDArray
            Gradient of the likelihood.
        """
        self.finite_difference_gradient_likelihood(eta, y, **kwargs)

    def finite_difference_gradient_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        h: float = 1e-4,
        **kwargs,
    ) -> NDArray:
        """Evaluate the finite difference gradient of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        h : float
            Finite difference step size.
        **kwargs : optional
            Hyperparameters for likelihood.

        Returns
        -------
        finite_difference_gradient : NDArray
            Finite difference gradient of the likelihood.

        Notes
        -----
        The Gradient of the likelihood is computed using a five-point stencil as follows:

        .. math:: \grad{f}=\frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h}
        """
        f1 = self.evaluate_likelihood(eta + h, y, **kwargs)
        f2 = self.evaluate_likelihood(eta + 2 * h, y, **kwargs)
        b1 = self.evaluate_likelihood(eta - h, y, **kwargs)
        b2 = self.evaluate_likelihood(eta - 2 * h, y, **kwargs)
        grad = (-f2 + 8 * f1 - 8 * b1 + b2) / (12 * h)
        return grad

    def evaluate_hessian_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        **kwargs,
    ) -> ArrayLike:
        """Evaluate the Hessian of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        **kwargs : optional
            Hyperparameters for likelihood.

        Returns
        -------
        hessian_likelihood : ArrayLike
            Hessian of the likelihood.
        """
        self.finite_difference_hessian_likelihood(eta, y, **kwargs)

    def finite_difference_hessian_likelihood(
        self,
        eta: NDArray,
        y: NDArray,
        h: float = 1e-2,
        **kwargs,
    ) -> NDArray:
        """Evaluate the finite difference Hessian of the likelihood wrt to eta = Ax.

        Parameters
        ----------
        eta : NDArray
            Vector of the linear predictor.
        y : NDArray
            Vector of the observations.
        h : float
            Finite difference step size.
        **kwargs : optional
            Hyperparameters for likelihood.

        Returns
        -------
        finite_difference_hessian : NDArray
            Finite difference hessian of the likelihood.

        Notes
        -----
        The Hessian of the likelihood is computed using a five-point stencil as follows:

        .. math:: \hess{f}=\frac{-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)}{12h^2}
        """
        f1 = self.evaluate_likelihood(eta + h, y, **kwargs)
        f2 = self.evaluate_likelihood(eta + 2 * h, y, **kwargs)
        b1 = self.evaluate_likelihood(eta - h, y, **kwargs)
        b2 = self.evaluate_likelihood(eta - 2 * h, y, **kwargs)
        c = self.evaluate_likelihood(eta, y, **kwargs)
        hessian = (-f2 + 16 * f1 - 30 * c + 16 * b1 - b2) / (12 * h * h)
        return hessian
