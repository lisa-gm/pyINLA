import numpy as np
import pytest
from scipy.sparse import diags

from pyinla.likelihoods.poisson import PoissonLikelihood
from pyinla.utils import (
    gradient_finite_difference_3pt,
    gradient_finite_difference_5pt,
    hessian_diag_finite_difference_3pt,
    hessian_diag_finite_difference_5pt,
)


@pytest.mark.parametrize("n_observations", [10, 100, 500])
def test_finite_difference_poisson(
    generate_poisson_data,
    n_observations: int,
    pyinla_config,
):
    eta, y, _, theta_likelihood = generate_poisson_data

    likelihood_instance = PoissonLikelihood(pyinla_config, n_observations)
    grad_likelihood_inla = likelihood_instance.evaluate_gradient_likelihood(
        eta, y, theta_likelihood
    )

    grad_3pt = gradient_finite_difference_3pt(
        likelihood_instance.evaluate_likelihood, eta, y, theta_likelihood
    )
    grad_5pt = gradient_finite_difference_5pt(
        likelihood_instance.evaluate_likelihood, eta, y, theta_likelihood
    )

    rtol = 1e-01
    atol = 1e-03

    assert np.allclose(grad_likelihood_inla, grad_3pt, rtol=rtol, atol=atol)
    assert np.allclose(grad_likelihood_inla, grad_5pt, rtol=rtol, atol=atol)

    hess_likelihood_inla = likelihood_instance.evaluate_hessian_likelihood(
        eta, y, theta_likelihood
    )
    hess_3pt = diags(
        hessian_diag_finite_difference_3pt(
            likelihood_instance.evaluate_likelihood, eta, y, theta_likelihood
        )
    )
    hess_5pt = diags(
        hessian_diag_finite_difference_5pt(
            likelihood_instance.evaluate_likelihood, eta, y, theta_likelihood
        )
    )

    assert np.allclose(
        hess_likelihood_inla.diagonal(), hess_3pt.diagonal(), rtol=rtol, atol=atol
    )
    assert np.allclose(
        hess_likelihood_inla.diagonal(), hess_5pt.diagonal(), rtol=rtol, atol=atol
    )
