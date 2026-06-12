import numpy as np
import scipy.sparse as sp

from .matrix import Matrix
from .dense import DenseMatrix
from .sparse import SparseMatrix
from dalia.backend.config import default_hw_target


class BStructMatrix(Matrix):
    """Block-structured matrix class.

    This class supports the general implementation of 1D and 2D block-structured
    matrices. Every-single block in this matrix is a Matrix object, which can a DenseMatrix, a
    SparseMatrix, or even another BStructMatrix (which allows for recursive block
    structures of arbitrary depth.).

    The block matrix is stored as a 1D or 2D numpy array of Matrix objects, where
    each entry corresponds to a block. Each block is either a :class:`Matrix` instance
    or None (indicating a zero block).

    Parameters
    ----------
    blocks : list | np.ndarray
        A 1D or 2D list or numpy array where each element is a Matrix object representing
        a block. The shape of the outer list/array defines the block structure of the
        matrix. Blocks can be DenseMatrix, SparseMatrix, or even BStructMatrix for
        nested structures. None can be used to indicate zero blocks.

    Notes
    -----
    - When constructing a block-structured matrix with zero blocks, the user must ensure
    that every block row and block column contains at least one non-zero block. This is
    necessary to define the dimensions of the matrix and to avoid ambiguity in shape and size.

    Examples
    --------
    Create a 2x2 block-structured matrix using a 2D list of blocks:
    >>> import numpy as np
    >>> import scipy as sp
    >>> from dalia.backend.datastructures import DenseMatrix, SparseMatrix, BStructMatrix

    >>> block_00 = DenseMatrix(np.array([[1, 2], [3, 4]]))
    >>> block_01 = SparseMatrix(sp.csr_matrix([[0, 5], [6, 0]]))
    >>> block_10 = DenseMatrix(np.array([[7, 8], [9, 10]]))
    >>> block_11 = SparseMatrix(sp.csr_matrix([[0, 11], [12, 0]]))

    >>> bstruct = BStructMatrix([[block_00, block_01], [block_10, block_11]])

    Create a 2x2 block-structured matrix with a zero block:
    >>> bstruct_zero = BStructMatrix([[block_00, None], [block_10, block_11]])
    """

    def __init__(self, blocks: list | np.ndarray, hw_target=default_hw_target):
        # Validate input is 1D or 2D list/array of Matrix objects or None
        if not isinstance(blocks, (list, np.ndarray)):
            raise TypeError(
                f"Blocks must be a list or numpy array, got {type(blocks).__name__}"
            )

        # Convert list to numpy array and check dimensions
        if isinstance(blocks, list):
            blocks = np.array(blocks, dtype=Matrix)
        if blocks.ndim not in (1, 2):
            raise ValueError(f"Blocks must be a 1D or 2D array, got {blocks.ndim}D")
        n_brows, n_bcols = blocks.shape if blocks.ndim == 2 else (blocks.size, 1)

        # Validate that each block is a Matrix object or None
        for i in range(n_brows):
            for j in range(n_bcols):
                block = blocks[i, j]
                if block is not None and not isinstance(block, Matrix):
                    raise TypeError(
                        f"Each block must be a Matrix object or None, got {type(block).__name__} at position ({i}, {j})"
                    )

        # Temporary storage for row/col sizes (to be filled during validation)
        row_sizes = [None] * n_brows
        col_sizes = [None] * n_bcols

        for i in range(n_brows):
            # Validate that every block row and block column contains at least one non-zero block
            if all(blocks[i, j] is None for j in range(n_bcols)):
                raise ValueError(f"Block row {i} cannot be all zero blocks")
            if all(blocks[j, i] is None for j in range(n_brows)):
                raise ValueError(f"Block column {i} cannot be all zero blocks")

            # Validate that all blocks in the same block row (resp. column) have the
            # same number of rows (resp. columns)
            row_block_shapes = [
                block._data.shape[0] for block in blocks[i, :] if block is not None
            ]
            if len(set(row_block_shapes)) > 1:
                raise ValueError(
                    f"All blocks in block row {i} must have the same number of rows, got {row_block_shapes}"
                )
            row_sizes[i] = row_block_shapes[
                0
            ]  # Store the common row size for this block row

            col_block_shapes = [
                block._data.shape[1] for block in blocks[:, i] if block is not None
            ]
            if len(set(col_block_shapes)) > 1:
                raise ValueError(
                    f"All blocks in block column {i} must have the same number of columns, got {col_block_shapes}"
                )
            col_sizes[i] = col_block_shapes[
                0
            ]  # Store the common column size for this block column

        # block-structure related attributes
        self._n_brows = n_brows
        self._n_bcols = n_bcols
        self._bshape = blocks.shape

        self._blocks = blocks

        # overall (value-wise) matrix attributes
        self._row_sizes = row_sizes
        self._col_sizes = col_sizes
        self._shape = (sum(row_sizes), sum(col_sizes))

        self._hw_target = hw_target
        self._propagate_hw_target()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def shape(self):
        """Overall shape of the block-structured matrix (total number of rows and columns)"""
        return self._shape

    @property
    def bshape(self):
        """Shape of the block structure (number of block rows and block columns)"""
        return self._bshape

    @property
    def dtype(self):
        """..."""
        raise NotImplementedError(
            "dtype property not implemented yet for BStructMatrix as it is considered for now as an undefined behavior."
        )

    @property
    def T(self):
        """Return the transpose of the block matrix.

        This could either be implemented as a view (if the underlying blocks support it.
        Would re-direct pointers in getitem to transposed relative blocks) or as a new
        BStructMatrix with transposed blocks and swapped block structure. For now, this
        is left unimplemented.
        """
        raise NotImplementedError("Transpose not implemented yet for BStructMatrix.")

    @property
    def hw_target(self):
        return self._hw_target

    @hw_target.setter
    def hw_target(self, target: str):
        if self._hw_target != target:
            self._hw_target = target
            self._propagate_hw_target()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        """Access a global element by (row, column) tuple.

        Note
        ----
        - Not implemented yet. See cross-blocks and fancy slicing drafts.
        """
        raise NotImplementedError(
            "Global element access not implemented yet for BStructMatrix. Use .toarray() for now."
        )

    def __setitem__(self, key, value):
        """Set a global element by (row, column) tuple.

        Note
        ----
        - Not implemented yet. See cross-blocks and fancy slicing drafts.
        """
        raise NotImplementedError(
            "Global element assignment not implemented yet for BStructMatrix. Use .toarray() for now."
        )

    def block(self, block_row: int, block_col: int) -> Matrix:
        """Access a specific block by its block row and block column indices."""
        if block_row < 0 or block_row >= self._n_brows:
            raise IndexError(f"Block row index {block_row} out of range")
        if block_col < 0 or block_col >= self._n_bcols:
            raise IndexError(f"Block column index {block_col} out of range")
        return self._blocks[block_row, block_col]

    # ------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------
    def to_dense(self) -> DenseMatrix:
        """Convert to a single dense matrix."""
        dense_array = np.zeros(self._shape, dtype=self._dtype)
        row_offset = 0
        for i, row_size in enumerate(self._row_sizes):
            col_offset = 0
            for j, col_size in enumerate(self._col_sizes):
                blk = self._blocks[i][j]
                if blk is not None:
                    dense_array[
                        row_offset : row_offset + row_size,
                        col_offset : col_offset + col_size,
                    ] = blk.toarray()
                col_offset += col_size
            row_offset += row_size
        return DenseMatrix(dense_array, hw_target=self._hw_target)

    def to_sparse(self) -> SparseMatrix:
        """Convert to a single sparse matrix (CSR)."""
        from scipy.sparse import coo_matrix

        rows, cols, data = [], [], []
        row_offset = 0
        for i, row_size in enumerate(self._row_sizes):
            col_offset = 0
            for j, col_size in enumerate(self._col_sizes):
                blk = self._blocks[i][j]
                if blk is not None:
                    # Get block as a sparse matrix in COO format
                    if isinstance(blk, SparseMatrix):
                        blk_sp = blk._data
                    elif isinstance(blk, DenseMatrix):
                        blk_sp = sp.csr_matrix(blk.toarray())
                    else:
                        # Recursively convert nested BStructMatrix
                        blk_sp = blk.to_sparse()._data
                    blk_coo = blk_sp.tocoo()
                    rows.extend(blk_coo.row + row_offset)
                    cols.extend(blk_coo.col + col_offset)
                    data.extend(blk_coo.data)
                col_offset += col_size
            row_offset += row_size
        coo = coo_matrix((data, (rows, cols)), shape=self._shape, dtype=self._dtype)
        return SparseMatrix(coo.tocsr(), hw_target=self._hw_target)

    def toarray(self) -> np.ndarray:
        """Return dense numpy array (compatible with parent interface)."""
        return self.to_dense().toarray()

    # ------------------------------------------------------------------
    # Other public methods
    # ------------------------------------------------------------------
    def copy(self):
        """Return a deep copy of the block matrix."""
        copied_blocks = [
            [blk.copy() if blk is not None else None for blk in row]
            for row in self._blocks
        ]
        return BStructMatrix(copied_blocks, hw_target=self._hw_target)

    def __repr__(self) -> str:
        """Return a string representation like a NumPy array, showing block types."""
        # Build a 2D list of block type names
        block_names = []
        max_len = 0
        for i in range(self._n_brows):
            row_names = []
            for j in range(self._n_bcols):
                blk = self._blocks[i][j]
                if blk is None:
                    name = "ZeroBlock"
                else:
                    # Use the class name, stripping module prefix if present
                    name = type(blk).__name__
                row_names.append(name)
                max_len = max(max_len, len(name))
            block_names.append(row_names)

        # Format each row with aligned columns, enclosed in brackets
        rows_str = []
        for i, row in enumerate(block_names):
            # Format each element with right alignment (like numpy's default)
            formatted_row = "[" + ", ".join(f"{name:>{max_len}}" for name in row) + "]"
            rows_str.append(formatted_row)

        # Join rows with newline and indentation
        inner = ",\n       ".join(rows_str)
        result = f"BStructMatrix([{inner}], shape={self._bshape}, dtype=Matrix)"
        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _propagate_hw_target(self):
        """Recursively move all blocks to the current hardware target."""
        for row in self._blocks:
            for blk in row:
                if blk is not None:
                    blk.hw_target = self._hw_target

    # ------------------------------------------------------------------
    # Override parent methods that use _data (prevent delegation)
    # ------------------------------------------------------------------
    def _wrap_result(self, data):
        """Override because BStructMatrix does not use _data."""
        # This method is only called by parent arithmetic; we override those methods,
        # so this should never be invoked. Raise an error for safety.
        raise NotImplementedError(
            "_wrap_result not implemented for BStructMatrix; all arithmetic is overridden."
        )

    # Block numpy array protocol to avoid implicit conversion
    __array__ = None
    __array_struct__ = None
    __array_interface__ = None
