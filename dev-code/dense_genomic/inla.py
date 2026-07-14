"""..."""

import numpy as np
from hyperparameter import HyperparameterManager

from .model import StatisticalModel


def marginal_log_likelihood_approximation(
    hp_values: np.ndarray,
    model: StatisticalModel,
    hpm: HyperparameterManager,
) -> float:
    """
    Compute the INLA marginal log-likelihood approximation.

    Returns the scalar f = conditional - prior - likelihood - prior_hyperparameters
    """
    Q_prior = model.assemble_prior_precision_matrix(
        hyperparameters=hpm.convert_array_to_dict(
            array=hp_values,
            include_fixed=True,
        )
    )
    A = model.assemble_design_matrix()

    # Conditional precision: Q_cond = Q_prior - θ * AᵀA
    Q_cond = ...  # depends on likelihood structure

    # Compute the 4 terms
    f_cond = compute_conditional_latent(Q_cond)
    f_prior = compute_prior_latent(Q_prior)
    f_lik = compute_likelihood(y, A, Q_cond)
    f_hp = compute_prior_hyperparameters(hp_named)

    f = f_cond - f_prior - f_lik - f_hp
    return f


def objective(
    hp_values: np.ndarray,
    model: StatisticalModel,
    hpm: HyperparameterManager,
) -> float:
    """Compute the INLA objective function at the evaluated
    hyperparameter values as well as its gradient (jacobian)
    with respect to the hyperparameters.

    The objective function is defined as the negative of the
    marginal log-likelihood approximation.
    """
    hpm.buffer_update(hp_values)

    fun = marginal_log_likelihood_approximation(
        hp_values=hp_values,
        model=model,
        hpm=hpm,
    )
    jac = finite_difference_gradient(
        objective_fn=marginal_log_likelihood_approximation,
        hp_values=hp_values,
        model=model,
        hpm=hpm,
    )

    return (fun, jac)


def finite_difference_gradient(
    objective_fn,
    hp_values: np.ndarray,
    model: StatisticalModel,
    hpm: HyperparameterManager,
    stencil: int = 3,
    h: float = 1e-5,
) -> np.ndarray:
    """
    Compute gradient using finite differences.

    stencil=3: central difference, O(h²) accuracy, 2N evaluations
    stencil=5: higher-order central, O(h⁴) accuracy, 4N evaluations
    """
    n = len(hp_values)
    grad = np.zeros(n)

    for i in range(n):
        if stencil == 3:
            # Central difference: (f(x+h) - f(x-h)) / (2h)
            hp_plus = hp_values.copy()
            hp_plus[i] += h
            hp_minus = hp_values.copy()
            hp_minus[i] -= h
            grad[i] = (
                objective_fn(hp_plus, model, hpm) - objective_fn(hp_minus, model, hpm)
            ) / (2 * h)
        elif stencil == 5:
            # Higher-order central: (-f(x+2h) + 8*f(x+h) - 8*f(x-h) + f(x-2h)) / (12h)
            ...

    return grad
