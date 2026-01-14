"""
Base class for linear solvers.

Each linear solvers should provide the following functionalities:
- Return the factorization, either Cholesky, LDL^T, or LU, based on the matrix properties.
- Solve a linear system Ax = b using the chosen factorization.
- Perform the selected-inversion of a matrix A given it's factors (matching the sparsity pattern of A).

"""
