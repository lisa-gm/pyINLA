# generate random sparse ymmetric positive definite matrix using scipy.sparse.random
# factorize the matrix using scipy.sparse.linalg.splu

from os import environ

#import numpy as np
#import scipy.__config__ as config
import scipy.sparse as sp
import scipy.sparse.linalg as spla

environ["OMP_NUM_THREADS"] = "16"


def generate_sparse_spd_matrix(n, density=0.01):
    # Generate a random sparse matrix
    A = sp.random(n, n, density=density, format="csr")

    # Make the matrix symmetric
    A = (A + A.T) / 2

    # Add n*I to make the matrix positive definite
    A += n * sp.eye(n)

    return A


def factorize_matrix(A):
    # Factorize the matrix using scipy.sparse.linalg.splu
    lu = spla.splu(A)
    return lu


if __name__ == "__main__":
    n = 1000  # Size of the matrix
    density = 0.01  # Density of the sparse matrix

    # print("SciPy configuration:")
    # config.show()

    # Generate a random sparse symmetric positive definite matrix
    A = generate_sparse_spd_matrix(n, density)

    print("Calling factorize_matrix now...")
    # Factorize the matrix
    lu = factorize_matrix(A)

    # Print some information about the factorization
    print("Factorization complete.")
    print("L shape:", lu.L.shape)
    print("U shape:", lu.U.shape)
