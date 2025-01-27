# compute 1-dimensional, i.e. element-wise finite differences of vector inputs

import numpy as np

from pyinla import ArrayLike


def gradient_finite_difference_3pt(f, x0: ArrayLike, *args, h: float = 1e-3):
    """
    Compute gradient using 3-point stencil finite differences wrt to x0

    Parameters
    ----------
    f: function
        function to be evaluated
    x0: ArrayLike
        evaluation point
    *args: tuple
        additional arguments to be passed to f
    h: float. default 1e-3
        stepsize.

    Returns
    -------
    gradient: ArrayLike
        array with 1D derivatives.
    """

    # Number of dimensions
    n = len(x0)

    # Initialize gradient vector
    grad = np.zeros(n)

    # Compute the gradient in each dimension
    for i in range(n):
        x_forward = np.array(x0, dtype=float)
        x_backward = np.array(x0, dtype=float)

        x_forward[i] += h  # x + h in the i-th direction
        x_backward[i] -= h  # x - h in the i-th direction

        # Three-point central difference formula
        grad[i] = (f(x_forward, *args) - f(x_backward, *args)) / (2 * h)

    return grad


def gradient_finite_difference_5pt(f, x0, *args, h=1e-3):
    # Number of dimensions
    n = len(x0)

    # Initialize gradient vector
    grad = np.zeros(n)

    # Compute the gradient in each dimension
    for i in range(n):
        x_forward_2h = np.array(x0, dtype=float)
        x_forward_h = np.array(x0, dtype=float)
        x_backward_h = np.array(x0, dtype=float)
        x_backward_2h = np.array(x0, dtype=float)

        # Adjust positions for each direction
        x_forward_2h[i] += 2 * h  # x + 2h in the i-th direction
        x_forward_h[i] += h  # x + h in the i-th direction
        x_backward_h[i] -= h  # x - h in the i-th direction
        x_backward_2h[i] -= 2 * h  # x - 2h in the i-th direction

        # Five-point stencil central difference formula
        grad[i] = (
            -f(x_forward_2h, *args)
            + 8 * f(x_forward_h, *args)
            - 8 * f(x_backward_h, *args)
            + f(x_backward_2h, *args)
        ) / (12 * h)

    return grad


def hessian_diag_finite_difference_3pt(f, x0, *args, h=1e-3):
    """
    Compute the diagonal of the Hessian matrix of a scalar function using a second-order finite difference scheme.

    Parameters
    ----------
    f : function
        The function whose Hessian diagonal we want to compute. Should take a numpy array as input.
    x0 : np.ndarray
        The point at which to evaluate the diagonal of the Hessian.
    *args : tuple
        Additional arguments to be passed to the function `f`.
    h : float, optional
        The step size for the finite difference approximation. Defaults to 1e-3.

    Returns
    -------
    hessian_diag : np.ndarray
        The diagonal elements of the Hessian matrix of f evaluated at x0.
    """
    # Number of dimensions
    n = len(x0)

    # Initialize vector for diagonal of Hessian
    hessian_diag = np.zeros(n)

    # Compute second-order finite differences for each diagonal element
    for i in range(n):
        x_forward = np.array(x0, dtype=float)
        x_backward = np.array(x0, dtype=float)

        x_forward[i] += h  # x + h in the i-th direction
        x_backward[i] -= h  # x - h in the i-th direction

        # Second-order central difference formula for the second derivative w.r.t. x_i
        hessian_diag[i] = (
            f(x_forward, *args) - 2 * f(x0, *args) + f(x_backward, *args)
        ) / (h**2)

    return hessian_diag


def hessian_diag_finite_difference_5pt(f, x0, *args, h=1e-3):
    """
    Compute the diagonal of the Hessian matrix of a scalar function using a five-point stencil finite difference scheme.

    Parameters
    ----------
    f:  function
        The function whose Hessian diagonal we want to compute. Should take a numpy array as input.
    x0: np.ndarray:
        The point at which to evaluate the diagonal of the Hessian.
    *args : tuple
        Additional arguments to be passed to the function `f`.
    h: float
        The step size for the finite difference approximation. Defaults to 1e-3. Careful this plays a big role!

    Returns
    -------
    hessian_diag: np.ndarray
        The diagonal elements of the Hessian matrix of f evaluated at x0.
    """
    # Number of dimensions
    n = len(x0)

    # Initialize vector for diagonal of Hessian
    hessian_diag = np.zeros(n)

    # Compute second-order finite differences for each diagonal element
    for i in range(n):
        x_forward_2h = np.array(x0, dtype=float)
        x_forward_h = np.array(x0, dtype=float)
        x_backward_h = np.array(x0, dtype=float)
        x_backward_2h = np.array(x0, dtype=float)

        # Adjust positions for each direction
        x_forward_2h[i] += 2 * h  # x + 2h in the i-th direction
        x_forward_h[i] += h  # x + h in the i-th direction
        x_backward_h[i] -= h  # x - h in the i-th direction
        x_backward_2h[i] -= 2 * h  # x - 2h in the i-th direction

        # Five-point stencil central difference formula for the second derivative w.r.t. x_i
        hessian_diag[i] = (
            -f(x_forward_2h, *args)
            + 16 * f(x_forward_h, *args)
            - 30 * f(x0, *args)
            + 16 * f(x_backward_h, *args)
            - f(x_backward_2h, *args)
        ) / (12 * h**2)

    return hessian_diag
