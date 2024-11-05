import numpy as np
from mpi4py import MPI
from scipy.sparse import csc_matrix, sparray

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


# make print function that only prints if MPI rank is 0
def print_mpi(*args, **kwargs):
    if comm_rank == 0:
        print(*args, **kwargs)


def read_sym_CSC(filename: str) -> sparray:
    """Read in lower triangular part of symmetric matrix that is stored in CSC format.

    Input
    ------
    filename: string. path to file.


    Return
    ------
    A : sparray. returns sparse lower triangular matrix

    """

    with open(filename, "r") as f:
        n = int(f.readline().strip())
        n = int(f.readline().strip())
        nnz = int(f.readline().strip())

        inner_indices = np.zeros(nnz, dtype=int)
        outer_index_ptr = np.zeros(n + 1, dtype=int)
        values = np.zeros(nnz, dtype=float)

        for i in range(nnz):
            inner_indices[i] = int(f.readline().strip())

        for i in range(n + 1):
            outer_index_ptr[i] = int(f.readline().strip())

        for i in range(nnz):
            values[i] = float(f.readline().strip())

    # Create the lower triangular CSC matrix
    A = csc_matrix((values, inner_indices, outer_index_ptr), shape=(n, n))

    return A


def read_CSC(filename):
    with open(filename, "r") as f:
        nrows = int(f.readline().strip())
        ncols = int(f.readline().strip())
        nnz = int(f.readline().strip())

        inner_indices = np.zeros(nnz, dtype=int)
        outer_index_ptr = np.zeros(ncols + 1, dtype=int)
        values = np.zeros(nnz, dtype=float)

        for i in range(nnz):
            inner_indices[i] = int(f.readline().strip())

        for i in range(ncols + 1):
            outer_index_ptr[i] = int(f.readline().strip())

        for i in range(nnz):
            values[i] = float(f.readline().strip())

    # Create CSC matrix
    A = csc_matrix((values, inner_indices, outer_index_ptr), shape=(nrows, ncols))

    return A
