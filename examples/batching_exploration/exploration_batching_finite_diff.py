import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import eye, csc_matrix
import time

import cupy as cp

SEED = 1234
np.random.seed(SEED)


def x2(x):
    # treat input column-wise
    x2 = np.multiply(x, x).sum(axis=0)
    return x2


def x2_gpu(x):
    # treat input column-wise
    x2 = cp.multiply(x, x).sum(axis=0)
    return x2


def grad_x2(x):
    grad = 2 * x
    return grad


def get_section_sizes(num_elements: int, num_sections: int) -> tuple:
    """Computes the number of un-evenly divided elements per section.

    Parameters
    ----------
    num_elements : int
        The total number of elements to divide.
    num_sections : int
        The number of sections to divide the elements into. Defaults to
        the number of MPI ranks.

    Returns
    -------
    section_sizes, effective_num_elements : tuple
        A tuple containing the sizes of each section and the effective
        number of elements after division.

    """
    quotient, remainder = divmod(num_elements, num_sections)
    section_sizes = remainder * [quotient + 1] + (num_sections - remainder) * [quotient]
    effective_num_elements = max(section_sizes) * num_sections
    return section_sizes, effective_num_elements


def get_batches(num_sections: int, max_batch_size: int) -> tuple:
    # Get list of batches to perform
    batches_sizes, _ = get_section_sizes(
        num_elements=num_sections,
        num_sections=num_sections // min(max_batch_size, num_sections),
    )
    batches_slices = np.cumsum(np.array([0] + batches_sizes))

    return batches_sizes, batches_slices


def gradient_finite_difference_3pt_greedy(f, x0: ArrayLike, *args, h: float = 1e-3):
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


def gradient_finite_difference_3pt(
    f, x0: ArrayLike, *args, h: float = 1e-3, max_batch_size: int = 1
):
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
    x0 = x0.flatten()
    n = len(x0)

    # Initialize gradient vector
    grad = np.zeros(n)

    batches_sizes, batches_slices = get_batches(n, max_batch_size)

    # Compute the gradient in each dimension
    # x_forward = np.empty_like(x0)
    x_forward = np.empty((n, max(batches_sizes)), dtype=x0.dtype)
    # x_backward = np.empty_like(x0)
    x_backward = np.empty((n, max(batches_sizes)), dtype=x0.dtype)

    I = np.identity(max(batches_sizes)) * h

    for b in range(len(batches_sizes)):
        x_forward[:, : batches_sizes[b]] = np.repeat(
            x0[:, np.newaxis], batches_sizes[b], axis=1
        )
        x_backward[:, : batches_sizes[b]] = np.repeat(
            x0[:, np.newaxis], batches_sizes[b], axis=1
        )

        x_forward[batches_slices[b] : batches_slices[b + 1], : batches_sizes[b]] += I[
            : batches_sizes[b], : batches_sizes[b]
        ]
        x_backward[batches_slices[b] : batches_slices[b + 1], : batches_sizes[b]] -= I[
            : batches_sizes[b], : batches_sizes[b]
        ]

        # Three-point central difference formula
        grad[batches_slices[b] : batches_slices[b + 1]] = (
            f(x_forward[:, : batches_sizes[b]], *args)
            - f(x_backward[:, : batches_sizes[b]], *args)
        ) / (2 * h)

    return grad


def gradient_finite_difference_3pt_gpu(
    f, x0: ArrayLike, *args, h: float = 1e-3, max_batch_size: int = 1
):
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
    x0 = cp.asarray(x0).flatten()
    n = len(x0)

    # Initialize gradient vector
    grad = cp.zeros(n)

    batches_sizes, batches_slices = get_batches(n, max_batch_size)

    # Compute the gradient in each dimension
    x_forward = cp.empty((n, max(batches_sizes)), dtype=x0.dtype)
    x_backward = cp.empty((n, max(batches_sizes)), dtype=x0.dtype)

    I = cp.identity(max(batches_sizes)) * h

    for b in range(len(batches_sizes)):
        x_forward[:, : batches_sizes[b]] = cp.repeat(
            x0[:, cp.newaxis], batches_sizes[b], axis=1
        )
        x_backward[:, : batches_sizes[b]] = cp.repeat(
            x0[:, cp.newaxis], batches_sizes[b], axis=1
        )

        x_forward[batches_slices[b] : batches_slices[b + 1], : batches_sizes[b]] += I[
            : batches_sizes[b], : batches_sizes[b]
        ]
        x_backward[batches_slices[b] : batches_slices[b + 1], : batches_sizes[b]] -= I[
            : batches_sizes[b], : batches_sizes[b]
        ]

        # Three-point central difference formula
        grad[batches_slices[b] : batches_slices[b + 1]] = (
            f(x_forward[:, : batches_sizes[b]], *args)
            - f(x_backward[:, : batches_sizes[b]], *args)
        ) / (2 * h)

    return cp.asnumpy(grad)


if __name__ == "__main__":
    n = 10000  # Size of the matrix

    batch_size_list = [1, 10, 100, 1000, 100000]
    time_list_greedy = np.zeros(len(batch_size_list))
    time_list_cpu = np.zeros(len(batch_size_list))
    time_list_gpu = np.zeros(len(batch_size_list))
    num_entries = np.zeros(len(batch_size_list))
    for i in range(len(batch_size_list)):
        num_entries[i] = n * batch_size_list[i]

    grad_fd_greedy = np.zeros(n)
    grad_fd = np.zeros(n)
    grad_fd_gpu = np.zeros(n)

    x0 = np.random.rand(n)

    # print("Time for computing finite difference: ", t_fd)
    grad_analytical = grad_x2(x0)

    for i in range(len(batch_size_list)):
        tic = time.perf_counter()
        grad_fd_greedy[:] = gradient_finite_difference_3pt_greedy(x2, x0)
        toc = time.perf_counter()
        time_list_greedy[i] = toc - tic

        t_fd = time.perf_counter()
        grad_fd[:] = gradient_finite_difference_3pt(
            x2, x0, max_batch_size=batch_size_list[i]
        )
        time_list_cpu[i] = time.perf_counter() - t_fd

        tic = time.perf_counter()
        grad_fd_gpu[:] = gradient_finite_difference_3pt_gpu(
            x2_gpu, x0, max_batch_size=batch_size_list[i]
        )
        toc = time.perf_counter()
        time_list_gpu[i] = toc - tic

        print(
            f"Batch size: {batch_size_list[i]}, norm greedy:  {np.linalg.norm(grad_fd_greedy - grad_analytical)}, norm cpu vectorized:  {np.linalg.norm(grad_fd - grad_analytical)}, norm gpu:  {np.linalg.norm(grad_fd_gpu - grad_analytical)}",
            flush=True,
        )

    print("Time list greedy:   ", time_list_greedy)
    print("Time list cpu:      ", time_list_cpu)
    print("Time list gpu:      ", time_list_gpu)
    print("Batch size list: ", batch_size_list)
    print("No entries per batch: ", num_entries)
