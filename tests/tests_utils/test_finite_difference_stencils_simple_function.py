# test different finite difference stencils for gradient and hessian computation

import numpy as np
import pytest

from pyinla import ArrayLike
from pyinla.utils.finite_difference_stencils import (
    gradient_finite_difference_3pt,
    gradient_finite_difference_5pt,
    hessian_diag_finite_difference_3pt,
    hessian_diag_finite_difference_5pt,
)


@pytest.mark.parametrize(
    "x0",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([-1.0, -2.0, 3.0]),
        np.array([0.5, 0.2, 0.8]),
        np.array([1.0]),  # 1D case
    ],
)
@pytest.mark.parametrize(
    "y",
    [
        2.0,
        1.0,
        0.5,
    ],
)
def test_finite_difference_simple_function(
    x0: ArrayLike,
    y: float,
):
    def f(x, y):
        return y * np.sum(x**2)

    rtol = 1e-05
    atol = 1e-08

    grad_analytical = 2 * y * x0
    grad_3pt = gradient_finite_difference_3pt(f, x0, y)
    grad_5pt = gradient_finite_difference_5pt(f, x0, y)

    assert np.allclose(grad_analytical, grad_3pt, rtol=rtol, atol=atol)
    assert np.allclose(grad_analytical, grad_5pt, rtol=rtol, atol=atol)

    hess_analytical = 2 * y * np.ones_like(x0)
    hess_3pt = hessian_diag_finite_difference_3pt(f, x0, y)
    hess_5pt = hessian_diag_finite_difference_5pt(f, x0, y)

    assert np.allclose(hess_analytical, hess_3pt, rtol=rtol, atol=atol)
    assert np.allclose(hess_analytical, hess_5pt, rtol=rtol, atol=atol)
