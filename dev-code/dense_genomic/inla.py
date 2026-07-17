"""...



Design decisions:

- We add a conditional inference stage to find the mode:
```python
mode = find_conditional_mode(
    q_prior,
    model,
    hp,
)
```
Which in turns allow to compute the lapalce correction as follows:
```python
f_cond = compute_laplace_correction(
    mode,
)
```

We suggest new names:
- conditional_latent_parameters(Q_cond) -> laplace_correction()
- prior_latent_parameters(Q_prior) -> log_latent_prior()
- likelihood(observations, A, Q_cond) -> log_likelihood()
- log_prior_hyperparameters(hp_dict) -> log_hyper_prior()

Separation of concerns:
Model knows:
- q_prior(theta)
- a
- y
(nothing about linalg)
Backend knows:
- factorize()
- solve()
- logdet()
(nothing about stats)
INLA knows:
- q_cond
- mode
- laplace_correction

This means that the Model might need to be able to assemble the information vector:
Model
-----
assemble_prior_precision()
assemble_information_vector()
# The design matrix is a property of the model and
# its assembly is independent of the hyperparameters
# or optimization related things
assemble_design_matrix() -> design_matrix()


```struct
objective(theta)
│
├── assemble_prior_precision(theta)
│
├── assemble_information_vector(a, observations)
│
├── find_conditional_mode(...)
│      │
│      ├── Gaussian
│      │      sparse solve
│      │
│      └── Non-Gaussian
│             Newton iterations
│
├── log_likelihood(mode)
│
├── log_latent_prior(mode)
│
├── log_hyper_prior(theta)
│
├── laplace_correction(mode)
│
└── return
      laplace
      - latent_prior
      - likelihood
      - hyper_prior
```

"""

from copy import deepcopy
from gc import collect

import numpy as np
from dev_utils import exit_as_expected, matshow_matrices
from hp_manager import HyperparameterManager
from model import StatisticalModel

from dalia.backend.blas.l2 import xxmv
from dalia.backend.blas.l3 import xxrk
from dalia.backend.datastructures import Matrix, Vector


def assemble_conditional_precision(q_prior: Matrix, a: Matrix, q_lik: Matrix = None):
    """
    Conditional precision matrix seems to be refere-able to as
    "likelihood precision matrix"?

    Q_cond = Q_prior + Aᵀ Q_lik A
    """
    # 1. q_cond = q_prior.copy()
    # . Matrix.copy() perform a deepcopy of the underlying `._data array`.
    q_cond: Matrix = q_prior.copy()

    # 2. q_cond += a.T @ q_lik @ a
    # . q_lik = identity because of Gaussian Likelihood, it can be ignored for now
    # . use xxrk (syrk) routine
    # . perform computation in-place on q_cond
    xxrk(
        uplo="l",
        trans_a="t",
        alpha=1.0,
        a=a,
        beta=1.0,
        c=q_cond,
        hw_target="default",
    )

    print(type(q_cond), q_cond.shape, q_cond.dtype, q_cond.hw_target)

    return q_cond


def assemble_information_vector(
    a: Matrix,
    q_lik: Matrix = None,
    observations: Vector = None,
):
    """Assemble the information vector for the conditional latent parameters.

    Parameters
    ----------
    a : Matrix
        The design matrix.
    q_lik : Matrix, optional
        The likelihood precision matrix, by default None.
    observations : Vector, optional
        The observed data, by default None.

    Maybe this should be part of the model, since it depends
    on the observations and the design matrix?
    -> Which are both properties of the model.
    -> This would depends on where Q_lik comes from, if it is part of the model or not.

    b = Aᵀ Q_lik y
    """
    # . q_lik = identity because of Gaussian Likelihood, it can be ignored for now
    # . use xxmv (symv) routine
    information_vector: Vector = xxmv(
        uplo="l",
        alpha=1.0,
        a=a,
        x=observations,
        beta=0.0,
        y=None,
        hw_target="default",
    )

    return information_vector


def find_conditional_mode(
    q_cond: Matrix,
    information_vector: Vector,
):
    """

    For Gaussian:
        0. Get Q_cond = Q_prior + AᵀQ_likA
        1. factorize Q_cond
        2. solve Q_cond * x_mode = AᵀQ_lik y

    For Non-Gaussian:
        0. Get Q_cond = Q_prior + AᵀQ_likA
        1. factorize Q_cond
        2. Newton iterations to find mode:
            x_mode^(k+1) = x_mode^(k) - H⁻¹ * g
            where H = Hessian of log-likelihood at x_mode^(k)
        3. return x_mode^(k+1) when convergence is reached
    """
    ...
    # backend.factorize(q_cond)
    # return backend.solve(information_vector)


def log_likelihood(mode): ...


def log_latent_prior(mode): ...


def log_hyper_prior(hp_dict): ...


def laplace_correction(l_cond, mode): ...


def negative_log_marginal_posterior(
    hp_dict: dict[str, float],
    model: StatisticalModel,
) -> float:
    """
    Compute the INLA marginal log-likelihood approximation.

    Returns the scalar f = conditional - prior - likelihood - prior_hyperparameters
    """
    # . assemble prior precision matrix
    q_prior: Matrix = model.assemble_prior_precision_matrix(
        hyperparameters_values=hp_dict
    )

    # . assemble conditional precision matrix
    q_cond: Matrix = assemble_conditional_precision(
        q_prior=q_prior,
        a=model.design_matrix(),
        # For now we know the likelihood to be Gaussian
        # q_lik=model.assemble_likelihood_precision_matrix(),
    )

    matshow_matrices(
        matrices=[q_prior.toarray(), q_cond.toarray()],
        titles=["q_prior", "q_cond"],
        plot_type="spy",
    )

    # Stop here for now, not implemented after...
    exit_as_expected()

    # . assemble information vector
    information_vector: Vector = assemble_information_vector(
        a=model.design_matrix(),
        # For now we know the likelihood to be Gaussian
        # q_lik=model.assemble_likelihood_precision_matrix(),
        observations=model.observations,
    )

    # matshow_matrices(
    #     matrices=[q_prior.toarray(), q_cond.toarray(), information_vector.toarray()],
    #     titles=["q_prior", "q_cond", "information_vector"],
    #     plot_type="spy",
    # )

    # # Stop here for now, not implemented after...
    # exit_as_expected()

    # . find_conditional_mode(q_prior, model, hp_dict)
    mode = find_conditional_mode(
        q_cond=q_cond,
        information_vector=information_vector,
    )

    # . compute the components of the INLA objective function
    f_likelihood = log_likelihood(mode=mode)
    f_log_latent_prior = log_latent_prior(mode=mode)
    f_log_hyper_prior = log_hyper_prior(hp_dict=hp_dict)
    f_laplace_correction = laplace_correction(l_cond=l_cond, mode=mode)

    # . assemble the final objective function value
    f = f_laplace_correction - f_log_latent_prior - f_likelihood - f_log_hyper_prior

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
    # . compute the marginal log-likelihood approximation at current points
    hp_dict: dict[str, float] = hpm.convert_array_to_dict(
        array=hp_values,
        include_fixed=True,
    )
    fun = negative_log_marginal_posterior(
        hp_dict=hp_dict,
        model=model,
    )

    # . compute Jacobian
    jac = finite_difference_gradient(
        objective_fn=negative_log_marginal_posterior,
        hp_values=hp_values,
        model=model,
        hpm=hpm,
    )

    # Cleanup
    # . Maybe it is a good idea to force garbage collection
    # between calls to the objective function?
    collect()

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
            # . f(x+h)
            hp_plus = hp_values.copy()
            hp_plus[i] += h
            hp_plus_dict: dict[str, float] = hpm.convert_array_to_dict(
                array=hp_plus,
                include_fixed=True,
            )
            # . f(x-h)
            hp_minus = hp_values.copy()
            hp_minus[i] -= h
            hp_minus_dict: dict[str, float] = hpm.convert_array_to_dict(
                array=hp_minus,
                include_fixed=True,
            )
            # . compute gradient
            grad[i] = (
                objective_fn(hp_plus_dict, model) - objective_fn(hp_minus_dict, model)
            ) / (2 * h)
        elif stencil == 5:
            # Higher-order central: (-f(x+2h) + 8*f(x+h) - 8*f(x-h) + f(x-2h)) / (12h)
            ...

    return grad
