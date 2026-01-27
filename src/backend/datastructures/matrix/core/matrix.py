"""
Design choices: Explicit sub-class for Sparse, Dense and Structured cases.

A Matrix class is the base class of all matrix types. There is no automatic dispatch
and the user need to create the subclass that matches the correct matrix type.
We opted for this coice as the sparsity is know from a statistical model implementation perspective.

The operatios are dispatched automatically based on the type of the operands.


"""

# src/backend/datastructures/matrix/core/matrix.py

from abc import ABC

from backend.datastructures.matrix.dispatch import blas_dispatch, Operation

from .utils import wrap_result, toarray


class Matrix(ABC):
    # 1. Class attributes (if any)
    __array_ufunc__ = None  # Disable numpy ufuncs to avoid conflicts

    # 2. Initialization
    def __init__(self, data):
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

    # @property
    # def shape(self):
    #     return self._data.shape

    # @property
    # def ndim(self):
    #     return self._data.ndim

    # 5. Comparison operators (if needed)

    # 6. Arithmetic operators (standard order)
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

    # 10. Public methods
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
        return wrap_result(data)
