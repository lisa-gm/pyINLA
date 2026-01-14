"""
Base class for precision matrices.

An implementation of this class should provide the following functionalities:
- CPU and GPU functionalities.
- Basic BLAS 2 and 3 operations:
    - Matrix-vector products.
    - Matrix-matrix products.
    - Triangular solves (functionality provided by the linear_solver class, get used here).
- Linear system related functionalities:
    - factorize
    - solve
    - compute log-det
    - selected-inversion / inverse
- Provide access to the matrix entries (if structured by blocks, provide access to the blocks?).
- Provide access to the matrix shape
- Provide I/O (seq. and dist.) functionalities

The sparsity pattern of the precision matrix is likely an attribute of the class:
- Is the matrix dense or sparse? Does it have a specific structure (banded, block-diagonal, etc.)?

Open questions:
- Is being a conditional or prior precision matrix a property of the class?

"""
