# src/dalia/backend/datastructures/matrix/core/matrix.py
from abc import ABC

from dalia.backend.datastructures.matrix.dispatch import Operation, blas_dispatch
from dalia.backend.config import default_hw_target

from .utils import toarray, wrap_result, tohost, toaccelerator, settarget


class Matrix(ABC):
    """Abstract base class for all matrix types in DALIA.

    Matrix provides a unified interface for dense and sparse matrix operations
    with automatic dispatch to optimized BLAS/LAPACK backends. This allows users
    to perform operations without worrying about the underlying matrix format or
    explicitly choosing between numpy and scipy operations.

    **Do not instantiate Matrix directly.** Use concrete subclasses:
    - :class:`DenseMatrix` for dense matrices (wraps numpy.ndarray)
    - :class:`SparseMatrix` for sparse matrices (wraps scipy.sparse in CSR format)

    Key Design Principles
    ---------------------
    1. **Type Consistency**: All operations return Matrix subclasses, never raw
       numpy/scipy arrays. This prevents accidental loss of type information.

    2. **Automatic Dispatch**: Operations automatically select optimal backends:
       - Dense x Dense → Dense
       - Sparse x Sparse → Sparse
       - Mixed operations → Depends on result sparsity

    3. **Explicit Conversion**: Matrix blocks numpy's array protocol to maintain
       type consistency. Use `.toarray()` for explicit conversion to numpy arrays.

    4. **External Compatibility**: Works seamlessly with scipy.sparse and numpy
       in mixed operations while preserving Matrix types in results.

    Choosing Subclasses
    -------------------
    - Use **SparseMatrix** for large matrices with mostly zero entries
      (e.g., precision matrices from SPDE models, design matrices with many zeros)
    - Use **DenseMatrix** for small or fully populated matrices
      (e.g., covariance matrices, small design matrices)
    - Use **BStructMatrix** for block-structured matrices.
      (e.g. precision matrices that results of the assembly of several sub-models)

    Common Operations
    -----------------
    Matrix multiplication (`@`), addition (`+`), subtraction (`-`), transpose (`.T`),
    and element access (`[]`) are supported. All operations return appropriate
    Matrix subclasses based on result sparsity.

    Attributes
    ----------
    _data : numpy.ndarray or scipy.sparse matrix
        Underlying matrix data (access via `._data` when needed)

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from backend.datastructures import DenseMatrix, SparseMatrix, BStructMatrix

    Create matrices:

    >>> dense = DenseMatrix(np.array([[1, 2], [3, 4]]))
    >>> sparse = SparseMatrix(sp.csr_matrix([[1, 0], [0, 2]]))
    >>> bstruct = BStructMatrix([[dense, sparse], [sparse, dense]])

    Operations return Matrix types:

    >>> result = dense @ sparse
    >>> isinstance(result, DenseMatrix)
    True

    Mixed operations with external types:

    >>> scipy_csr = sp.csr_matrix([[1, 0], [0, 1]])
    >>> result = scipy_csr @ dense  # Returns DenseMatrix, not numpy array
    >>> type(result)
    <class 'backend.datastructures.matrix.core.dense.DenseMatrix'>

    Explicit conversion when needed:

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(sparse.toarray())  # Convert for plotting

    Copy matrices:

    >>> copy = dense.copy()  # Independent copy
    >>> copy[0, 0] = 999
    >>> dense[0, 0]  # Original unchanged
    1

    Transpose:

    >>> transposed = dense.T
    >>> transposed.shape
    (2, 2)

    See Also
    --------
    DenseMatrix : Concrete class for dense matrices
    SparseMatrix : Concrete class for sparse matrices (CSR format)
    BStructMatrix : Block-structured matrix class for hierarchical models

    Notes
    -----
    - Matrix blocks the numpy array protocol (`__array__`, `__array_interface__`,
    `__array_struct__`) to prevent implicit conversion. This ensures that
    operations like `scipy.sparse @ Matrix` properly return Matrix types
    rather than unwrapped numpy arrays. This design follows pandas/PyTorch
    patterns rather than numpy's implicit conversion approach.
    """

    # 1. Class attributes (if any)
    __array_ufunc__ = None  # Disable numpy ufuncs to avoid conflicts

    # 2. Initialization
    def __init__(self, data, hw_target=default_hw_target):

        if data.dtype.char not in "fdFD":
            raise TypeError(
                f"Unsupported data type '{data.dtype}'. Only float32 and float64 are supported."
            )

        data, hw_target = settarget(data, hw_target)

        self._hw_target = hw_target
        self._data = data

    # 3. Special representation methods

    # 4. Properties (grouped together)
    @property
    def T(self):
        """Transpose of the matrix

        For dense matrices: Returns a view (no copy, behaves like numpy.ndarray)
        For sparse matrices: Returns a new matrix (copy, behaves like scipy.sparse)
        """
        # pylint: disable=invalid-name
        return wrap_result(self._data.T)

    @property
    def hw_target(self):
        """Hardware where the matrix data is stored ('host' or 'accelerator')"""
        return self._hw_target

    @hw_target.setter
    def hw_target(self, hw_target):
        """Set hardware target for the matrix data

        Args:
            hw_target (str): 'host' or if supported by the system: 'accelerator'.
        """
        if self._hw_target != hw_target:
            self._data, self._hw_target = settarget(self._data, hw_target)

    # 5. Comparison operators (if needed)

    # 6. Arithmetic operators (standard order)
    def __mul__(self, other):
        other_data = other._data if isinstance(other, Matrix) else other
        result_data = blas_dispatch(Operation.MUL, self._data, other_data)
        return self._wrap_result(
            result_data
        )

    def __matmul__(self, other):
        other_data = other._data if isinstance(other, Matrix) else other
        result_data = blas_dispatch(Operation.MATMUL, self._data, other_data)
        return self._wrap_result(
            result_data
        )  # Wrapping to correct Matrix() happens here

    def __add__(self, other):
        other_data = other._data if isinstance(other, Matrix) else other
        result_data = blas_dispatch(Operation.ADD, self._data, other_data)
        return self._wrap_result(result_data)

    def __sub__(self, other):
        other_data = other._data if isinstance(other, Matrix) else other
        result_data = blas_dispatch(Operation.SUB, self._data, other_data)
        return self._wrap_result(result_data)

    # 7. Right-hand operators (same order as above)
    def __rmatmul__(self, other):
        """Right-hand matrix multiplication: other @ self"""
        # other is the left operand (likely numpy/scipy, not wrapped)
        # self is the right operand (our Matrix)
        other_data = other._data if isinstance(other, Matrix) else other
        result_data = blas_dispatch(Operation.MATMUL, other_data, self._data)
        return self._wrap_result(result_data)

    def __radd__(self, other):
        """Right-hand addition: other + self"""
        other_data = other._data if isinstance(other, Matrix) else other
        result_data = blas_dispatch(Operation.ADD, other_data, self._data)
        return self._wrap_result(result_data)

    def __rsub__(self, other):
        """Right-hand subtraction: other - self"""
        other_data = other._data if isinstance(other, Matrix) else other
        result_data = blas_dispatch(Operation.SUB, other_data, self._data)
        return self._wrap_result(result_data)

    # 8. In-place operators (if supported)

    # 9. Other special methods
    def __getattr__(self, name):
        """Delegate attribute access to underlying data, blocking array protocol.

        Provides transparent access to properties (.shape, .dtype) and methods
        (.sum(), .mean()) from the underlying _data object.

        Blocks array protocol attributes to prevent implicit conversion:
        - __array__: Used by np.asarray() and np.asanyarray()
        - __array_struct__: Provides direct memory access via buffer protocol
        - __array_interface__: Dict-based array protocol for memory sharing

        Without blocking these, scipy/numpy would bypass __rmatmul__, __radd__, etc.

        Args:
            name: Attribute name

        Returns:
            Attribute from self._data

        Raises:
            AttributeError: If attribute doesn't exist or is blocked
        """
        if name in ("__array__", "__array_struct__", "__array_interface__"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                f"Use .toarray() for explicit conversion."
            )
        return getattr(self._data, name)

    def __getitem__(self, key):
        """Access matrix elements"""
        return self._data[key]

    def __setitem__(self, key, value):
        """Set matrix elements"""
        self._data[key] = value

    def __repr__(self):
        return self._data.__repr__()

    # 10. Public methods
    def copy(self):
        """Create a deep copy of the matrix.

        Returns a new Matrix instance with an independent copy of the underlying
        data. Modifications to the copy will not affect the original matrix.

        Returns
        -------
        Matrix
            New instance of the same Matrix subclass (SparseMatrix or DenseMatrix)
            with copied data.

        Examples
        --------
        >>> original = SparseMatrix(sp.csr_matrix([[1, 0], [0, 2]]))
        >>> duplicate = original.copy()
        >>> duplicate is original
        False
        >>> duplicate._data is original._data
        False
        >>> duplicate[0, 0] = 999  # Does not affect original
        """
        return type(self)(self._data.copy())

    def toarray(self):
        """Convert Matrix to dense numpy array (explicit conversion).

        This method provides explicit conversion to numpy.ndarray, which is
        required because Matrix blocks implicit conversion via the array protocol
        (__array__, __array_struct__, __array_interface__).

        Design rationale:
            Matrix prioritizes type consistency over implicit compatibility.
            By blocking implicit conversion, we ensure:
            - scipy.sparse @ Matrix → Matrix (not numpy.ndarray)
            - numpy.ndarray + Matrix → Matrix (not numpy.ndarray)
            - All operations return properly wrapped Matrix types

            This follows the pandas/PyTorch pattern where explicit conversion
            is required (.to_numpy(), .numpy()) rather than the numpy pattern
            where np.asarray() works implicitly.

        When to use:
            - Interfacing with libraries expecting raw numpy arrays
            - Plotting: plt.imshow(matrix.toarray())
            - Serialization: np.save('file', matrix.toarray())
            - Passing to numpy functions that don't work with Matrix

        Returns:
            numpy.ndarray: Dense 2D array, regardless of source sparsity.
                          For sparse matrices, this converts to dense (copy).
                          For dense matrices, this may return a view.

        See Also:
            ._data: Access underlying data object directly (numpy or scipy)

        Examples:
            >>> sparse = SparseMatrix(scipy.sparse.csr_matrix([[1, 0], [0, 2]]))
            >>> sparse.toarray()
            array([[1, 0],
                   [0, 2]])

            >>> dense = DenseMatrix(np.array([[1, 2]]))
            >>> dense.toarray()
            array([[1, 2]])
        """
        return toarray(self._data)

    # 11. Private/protected methods (start with _)
    def _wrap_result(self, data):
        """Wrap the result data in the appropriate Matrix subclass"""
        return wrap_result(data, hw_target=self._hw_target)
