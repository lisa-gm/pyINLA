# The Model

### Overview

This example fits a Gaussian likelihood model with an AR(1) latent process and one fixed effect (an intercept). The data consists of n = 1000 observations generated as y = eta + noise, where eta = u + intercept, u is a zero mean AR(1) process with autocorrelation phi = 0.9 and marginal variance s2 = 5 (precision tau = 1/s2), and the observation noise is Gaussian with precision 100.

The model contains three hyperparameters, theta = (phi, tau, prec_o):

1. phi: autocorrelation of the AR(1) process, constrained to (0, 1), with a Beta(5, 1) prior.
2. tau: precision of the AR(1) process, constrained to be positive, with a Gamma(2, 1) prior.
3. prec_o: precision of the Gaussian observation noise, with a Gaussian prior centered at the true value.

The latent field has dimension 1001: the 1000 AR(1) states plus the intercept. The fixed effect uses a weak prior precision of 0.001.

### Scripts

`generate_data.py` builds the tridiagonal AR(1) precision matrix Q, samples the latent field u through a Cholesky factorization of Q, adds the intercept (value 2) and Gaussian observation noise, and writes all files needed by the model: the design matrices `inputs_ar1/a.npz` and `inputs_regression/a.npz`, the observations `y.npy`, and the reference values `reference_outputs/theta_original.npy` and `reference_outputs/x_original.npy`. It also solves the conditional system directly as a sanity check of the generated data.

`run.py` loads the generated inputs and the reference outputs, constructs the model from an `AR1SubModel` and a `RegressionSubModel` combined with a Gaussian likelihood, and runs the full DALIA inference with a dense solver. It prints the estimated hyperparameters next to the true values, the covariance of theta, the mean of the fixed effect, error norms of the reconstructed linear predictor eta, a comparison of the marginal variances of the latent parameters against a dense reference, and quantiles of the marginal posterior of phi. It also plots the prior of tau and the marginal posterior distributions of the hyperparameters.

### Usage

Generate the data first, then run the inference:

```bash
python generate_data.py
python run.py
```

Both scripts resolve paths relative to their own location, so they can be executed from any directory.
