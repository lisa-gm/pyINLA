# src/dalia/backend/datastructures/matrix/core/sparse.py
import numpy as np
import scipy.sparse as sp

from dalia.backend.config import cupy_version

if cupy_version is not None:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sp

from .matrix import Matrix


class SparseMatrix(Matrix):
    """Sparse matrix wrapper with CSR canonical format.

    Wraps scipy.sparse matrices and ensures internal representation is CSR format
    for consistent performance characteristics. All sparse matrices are converted
    to canonical format (sorted indices, no duplicates) upon initialization.

    Parameters
    ----------
    data : scipy.sparse matrix
        Any scipy sparse matrix format (csr, csc, coo, etc.).
        Will be converted to CSR format internally and canonicalized.

    Raises
    ------
    TypeError
        If data is a dense numpy array, Matrix object, or unsupported type.
        Use .copy() method to duplicate an existing Matrix.

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> coo = sp.coo_matrix([[1, 0], [0, 2]])
    >>> sparse = SparseMatrix(coo)  # Converts to CSR internally
    >>> isinstance(sparse._data, sp.csr_matrix)
    True

    >>> # To duplicate a matrix, use .copy()
    >>> sparse2 = sparse.copy()  # Creates independent copy
    >>> sparse2 is sparse
    False
    >>> sparse2._data is sparse._data
    False
    """

    # 1. Class attributes (if any)

    # 2. Initialization
    def __init__(self, data, hw_target=None):
        # Reject Matrix objects - use .copy() method instead
        if isinstance(data, Matrix):
            raise TypeError(
                "Cannot create SparseMatrix from another Matrix object. "
                "Use matrix.copy() to duplicate a matrix."
            )

        # Reject dense arrays - user must be explicit
        if isinstance(data, np.ndarray):
            raise TypeError(
                "Cannot create SparseMatrix from dense numpy array. "
                "Use DenseMatrix instead, or convert to sparse format first with "
                "scipy.sparse.csr_matrix(array)."
            )

        if cupy_version is not None:
            if isinstance(data, cp.ndarray):
                raise TypeError(
                    "Cannot create SparseMatrix from dense cupy array. "
                    "Use DenseMatrix instead, or convert to sparse format first with "
                    "cupyx.scipy.sparse.csr_matrix(array)."
                )

        # Validate input is sparse
        if not (sp.issparse(data)):

            if cupy_version is not None:
                if not cu_sp.issparse(data):
                    raise TypeError(
                        f"SparseMatrix requires either scipy.sparse or cupyx.scipy.sparse matrix, got {type(data).__name__}"
                    )
            else:
                raise TypeError(
                    f"SparseMatrix requires scipy.sparse matrix, got {type(data).__name__}"
                )

        # Convert to canonical CSR format if needed
        if not isinstance(data, sp.csr_matrix):

            if cupy_version is not None:
                if not isinstance(data, cu_sp.csr_matrix):
                    data = data.tocsr()
            else:
                data = data.tocsr()

        # Ensure canonical format for optimal performance
        # This sorts indices and removes duplicates if needed
        if not data.has_canonical_format:
            data.sum_duplicates()
            data.sort_indices()

        # Initialize parent with CSR matrix
        super().__init__(data, hw_target)

    # 3. Special representation methods
    # 4. Properties (grouped together)
    # 5. Comparison operators (if needed)
    # 6. Arithmetic operators (standard order)
    # 7. Right-hand operators (same order as above)
    # 8. In-place operators (if supported)
    # 9. Other special methods
    # 10. Public methods
    # 11. Private/protected methods (start with _)
