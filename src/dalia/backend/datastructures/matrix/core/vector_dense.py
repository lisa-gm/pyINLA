# src/dalia/backend/datastructures/matrix/core/dense.py
import scipy.sparse as sp

from .dense import DenseMatrix


class Vector(DenseMatrix):
    """Dense vector wrapper.

    Wraps numpy ndarray as a vector.
    Allows vectors have two dimension > 1.

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
    >>> array = np.array([1, 2, 3])
    >>> vector = Vector(array)
    >>> isinstance(vector._data, np.ndarray)
    True

    >>> # To duplicate a vector, use .copy()
    >>> vector2 = vector.copy()  # Creates independent copy
    >>> vector2 is vector
    False
    >>> vector2._data is vector._data
    False
    """

    # 1. Class attributes (if any)

    # 2. Initialization
    def __init__(self, data, hw_target=None):
        # Initialize parent with dense array
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
