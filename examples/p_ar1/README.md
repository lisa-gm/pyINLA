# The Model

### Overview

This example fits a Poisson likelihood model with an AR(1) latent process and one fixed effect (an intercept). The data consists of n = 1000 counts generated as y ~ Poisson(E * exp(eta)), where eta = u + intercept, u is a zero mean AR(1) process with autocorrelation phi = 0.9 and marginal variance s2 = 1 (precision tau = 1/s2), and E is an exposure vector with entries sampled from {1, 2, 3}.

The model contains two hyperparameters, theta = (phi, tau):

1. phi: autocorrelation of the AR(1) process, constrained to (0, 1), with a Beta(5, 1) prior.
2. tau: precision of the AR(1) process, constrained to be positive, with a Gamma(2, 0.5) prior.

The Poisson likelihood has no hyperparameter of its own. The latent field has dimension 1001: the 1000 AR(1) states plus the intercept. The fixed effect uses a weak prior precision of 0.001.

### Scripts

`generate_data.py` builds the tridiagonal AR(1) precision matrix Q, samples the latent field u from the corresponding multivariate normal distribution, adds the intercept (value 2), draws the exposures E and the Poisson counts y, and writes all files needed by the model: the design matrices `inputs_ar1/a.npz` and `inputs_regression/a.npz`, the observations `y.npy`, the exposures `e.npy`, and the reference values `reference_outputs/theta_original.npy` and `reference_outputs/x_original.npy`.

`run.py` loads the generated inputs and the reference outputs, constructs the model from an `AR1SubModel` and a `RegressionSubModel` combined with a Poisson likelihood, and runs the hyperparameter optimization with a dense solver through `dalia.minimize()`. Initial values are set away from the truth (phi = 0.45, tau = 0.5). It prints the estimated hyperparameters next to the true values and the error norms of the recovered latent field.

### Usage

Generate the data first, then run the inference:

```bash
python generate_data.py
python run.py
```

`generate_data.py` writes its output directories relative to the current working directory while `run.py` reads them relative to the script location, so execute both from inside the `p_ar1` directory.
