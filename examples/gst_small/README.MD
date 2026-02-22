# Gaussian likelihood, Regression + Spatio-Temporal submodel example

This example contain a small spatio-temporal model.

--- Model ---
    n_hyperparameters: 4
    n_latent_parameters: 466
    n_observations: 4600
    likelihood: gaussian

--- SpatioTemporalSubModel ---
    ns: 92
    nt: 5
    manifold: plane

--- RegressionSubModel ---
    n_fixed_effects: 6
    fixed_effects_prior_precision: 0.001
