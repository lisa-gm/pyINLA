"""..."""


def objective():
    # Define the objective function for the INLA optimization.
    # . need to caracterize the architectural difference differences between getting forward difference gradient (and objective function at current hp) and auto-differentiation.
    # . . in particulat objective(), jacobian(), and how they are plugged intot he optimize()

    # Conditional of the latent field
    # . In the Gaussian case:
    # . . Q_cond = Q_prior - theta_likelihood * a.T @ a

    conditional_latent_parameters: float = ...

    prior_latent_parameters: float = ...
    likelihood: float = ...
    prior_hyperparameters: float = ...

    f = (
        conditional_latent_parameters
        - prior_latent_parameters
        - likelihood
        - prior_hyperparameters
    )

    return f


def jacobian():
    ...
    # return grad separatly? Need to check scipy.minimize documentation for this.
