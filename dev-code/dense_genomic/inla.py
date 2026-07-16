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
- conditional_latent_parameters -> log_conditional_gaussian()
- prior_latent_parameters -> log_latent_prior()
- likelihood -> log_likelihood()
- log_prior_hyperparameters -> log_hyper_prior()

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
assemble_design_matrix()


```struct
objective(theta)
│
├── assemble_prior_precision(theta)
│
├── assemble_likelihood(theta)
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

import matplotlib.pyplot as plt
import numpy as np
from hp_manager import HyperparameterManager
from model import StatisticalModel

from gc import collect

from dalia.backend.datastructures import Matrix, Vector

from dev_utils import exit_as_expected

# print("A:", A)
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# axs[0].matshow(q_prior.data, cmap="viridis")
# axs[0].set_title("Prior Precision Matrix")
# plt.colorbar(axs[0].matshow(q_prior.data, cmap="viridis"), ax=axs[0])
# axs[1].matshow(A.data, cmap="viridis")
# axs[1].set_title("Design Matrix")
# plt.colorbar(axs[1].matshow(A.data, cmap="viridis"), ax=axs[1])
# plt.show()

def find_conditional_mode(
        q_cond: Matrix, 
        information_vector: Vector,
    ):
    """
    
    For Gaussian:
        1. assemble Q_cond = Q_prior + AᵀQ_likA
        2. factorize Q_cond
        3. solve Q_cond * x_mode = AᵀQ_lik y

    For Non-Gaussian:
        1. assemble Q_cond = Q_prior + AᵀQ_likA
        2. factorize Q_cond
        3. Newton iterations to find mode:
            x_mode^(k+1) = x_mode^(k) - H⁻¹ * g
            where H = Hessian of log-likelihood at x_mode^(k)
        4. return x_mode^(k+1) when convergence is reached
    """
    ...

def assemble_conditional_precision(
        q_prior: Matrix,
        a: Matrix,
        q_lik: Matrix = None
    ):
    """
    Q_cond = Q_prior + Aᵀ Q_lik A
    """
    ...

def assemble_information_vector(
        a: Matrix,
        q_lik: Matrix = None,
        observations: Vector = None,
    ):
    """
    b = Aᵀ Q_lik y
    """
    ...

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

    # Stop here for now, not implemented after...
    exit_as_expected()

    # . assemble conditional precision matrix
    q_cond: Matrix = assemble_conditional_precision(
        q_prior=q_prior,
        a=model.design_matrix(),
        # For now we know the likelihood to be Gaussian
        # q_lik=model.assemble_likelihood_precision_matrix(),
    )

    # . assemble information vector
    information_vector: Vector = assemble_information_vector(
        a=model.design_matrix(),
        # For now we know the likelihood to be Gaussian
        # q_lik=model.assemble_likelihood_precision_matrix(),
        observations=model.observations,
    )

    # . assemble_likelihood(theta)
    ...

    # . find_conditional_mode(q_prior, model, hp_dict)
    mode = find_conditional_mode(
        q_cond=q_cond,
        information_vector=information_vector,
    )



    # f_cond = compute_laplace_correction(
    #     mode,
    # )

    # Compute the 4 terms
    f_cond = compute_conditional_latent(Q_cond)
    f_prior = compute_prior_latent(Q_prior)
    f_lik = compute_likelihood(model.observations, A, Q_cond)
    f_hp = compute_prior_hyperparameters(hp_dict)

    f = f_cond - f_prior - f_lik - f_hp
    return f

# def laplace_log_marginal_posterior()
# def inla_log_posterior()
# def negative_log_marginal_posterior(theta, data):
#     x_mode = latent_mode(theta, data)  # mode of the latent field
#     log_lik = log_likelihood(x_mode, theta, data)
#     log_latent = log_latent_prior(x_mode, theta)
#     log_hyper = log_hyper_prior(theta)
#     log_corr = laplace_correction(x_mode, theta, data)
#     return log_lik + log_latent + log_hyper - log_corr



def compute_conditional_latent(Q_cond): ...


def compute_prior_latent(Q_prior): ...


def compute_likelihood(observations, A, Q_cond): ...


def compute_prior_hyperparameters(hp_dict): ...


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
