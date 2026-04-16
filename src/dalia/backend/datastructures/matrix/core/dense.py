# src/dalia/backend/datastructures/matrix/core/dense.py
import scipy.sparse as sp

from .matrix import Matrix


class DenseMatrix(Matrix):
    """Dense matrix wrapper.

    Wraps numpy ndarray as a dense matrix.

    Parameters
    ----------
    data : numpy.ndarray
        Any numpy dense array.

    Raises
    ------
    TypeError
        If data is a sparse matrix or Matrix object.
        Use .copy() method to duplicate an existing Matrix.

    Examples
    --------
    >>> import numpy as np
    >>> array = np.array([[1, 0], [0, 2]])
    >>> dense = DenseMatrix(array)
    >>> isinstance(dense._data, np.ndarray)
    True

    >>> # To duplicate a matrix, use .copy()
    >>> dense2 = dense.copy()  # Creates independent copy
    >>> dense2 is dense
    False
    >>> dense2._data is dense._data
    False
    """

    # 1. Class attributes (if any)

    # 2. Initialization
    def __init__(self, data, device=None):
        # Reject Matrix objects - use .copy() method instead
        if isinstance(data, Matrix):
            raise TypeError(
                "Cannot create DenseMatrix from another Matrix object. "
                "Use matrix.copy() to duplicate a matrix."
            )

        # Reject sparse arrays - user must be explicit
        if sp.issparse(data):
            raise TypeError(
                "Cannot create DenseMatrix from sparse matrix. "
                "Use SparseMatrix instead, or convert to dense format first with "
                "sparse_matrix.toarray()."
            )

        # Initialize parent with dense array
        super().__init__(data, device)

    # 3. Special representation methods
    # 4. Properties (grouped together)
    # 5. Comparison operators (if needed)
    # 6. Arithmetic operators (standard order)
    # 7. Right-hand operators (same order as above)
    # 8. In-place operators (if supported)
    # 9. Other special methods
    # 10. Public methods
    # 11. Private/protected methods (start with _)
