import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import eye


def x2(x):
    return x.dot(x)


def gradient_finite_difference_3pt(f, x0: ArrayLike, *args, h: float = 1e-3):
    """
    Compute gradient using 3-point stencil finite differences wrt to x0

    Notes
    -----

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

    # # Compute the gradient in each dimension
    # for i in range(n):
    #     x_forward = np.array(x0, dtype=float)
    #     x_backward = np.array(x0, dtype=float)

    #     x_forward[i] += h  # x + h in the i-th direction
    #     x_backward[i] -= h  # x - h in the i-th direction

    #     # Three-point central difference formula
    #     grad[i] = (f(x_forward, *args) - f(x_backward, *args)) / (2 * h)

    # Compute the gradient in each dimension

    # replicate x0 column-wise
    batch_size = 1000

    x_mat = np.tile(x0, (n, min(batch_size, n))).T
    speye = h * eye(n, n, k=0)

    print(x_mat)

    num_batches = n // batch_size
    for i in range(num_batches):
        # compute index
        start_index = i * batch_size

        speye_slice = speye[:, start_index : start_index + batch_size]
        x_forward = x_mat + speye_slice
        #x_backward = x_mat - speye_slice
        print(x_forward)

        # Three-point central difference formula
        print(f(x_forward, *args))
        # grad[start_index:start_index + batch_size] = (f(x_forward, *args) - f(x_backward, *args)) / (2 * h)

    return grad


if __name__ == "__main__":
    n = 1000  # Size of the matrix
    density = 0.01  # Density of the sparse matrix

    x0 = np.random.rand(n)

    gradient_finite_difference_3pt(x2, x0)
