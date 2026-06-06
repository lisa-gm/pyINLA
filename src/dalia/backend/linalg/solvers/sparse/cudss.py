# src/dalia/backend/linalg/solvers/sparse/cudss.py
from .sparse_solver import SparseSolver
from dalia.backend.config import cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp
if nvmath_version is not None:
    import nvmath.bindings.cudss as nm

class CuDSS(SparseSolver):
    """CUDA Sparse Solver (cuDSS) for solving sparse linear systems.

    Uses NVIDIA's cuDSS library for GPU-accelerated sparse direct solving.
    Supports sparse matrices stored in CSR format on GPU memory.
    """

    # 1. Class attributes (if any)
    # 2. Initialization

    def __init__(self, matrix, overwrite_matrix: bool = False):
        """Initialize CuDSS solver.

        Parameters
        ----------
        matrix : SparseMatrix
            Sparse matrix in CSR format on GPU.
        overwrite_matrix : bool, optional
            If True, allows in-place factorization. Default is False.
        """
        super().__init__(matrix, overwrite_matrix)
        self._handle = None
        self._config = None
        self._data = None

    # 3. Special representation methods
    # 4. Properties (grouped together)
    # 5. Comparison operators (if needed)
    # 6. Arithmetic operators (standard order)
    # 7. Right-hand operators (same order as above)
    # 8. In-place operators (if supported)
    # 9. Other special methods

    def __del__(self):
        """Clean up cuDSS resources."""
        self._cleanup_cudss_objects()

    # 10. Public methods
    # 11. Private/protected methods (start with _)

    def _cleanup_cudss_objects(self):
        """Destroy all cuDSS objects."""
        if self._data is not None:
            try:
                nm.data_destroy(self._handle, self._data)
            except Exception:
                pass
            self._data = None

        if self._config is not None:
            try:
                nm.config_destroy(self._config)
            except Exception:
                pass
            self._config = None

        if self._handle is not None:
            try:
                nm.destroy(self._handle)
            except Exception:
                pass
            self._handle = None

    def _compute_factorization(self, overwrite: bool = False):
        """Compute LU factorization using cuDSS.

        The factorization is done during the solve phase in cuDSS,
        so this method just initializes cuDSS objects.

        Parameters
        ----------
        overwrite : bool, optional
            Unused for cuDSS. Default is False.

        Returns
        -------
        None
            Factorization is stored internally in cuDSS data object.
        """
        # Clean up any existing objects
        self._cleanup_cudss_objects()

        # Create cuDSS handle, config, and data objects
        self._handle = nm.create()
        self._config = nm.config_create()
        self._data = nm.data_create(self._handle)

        return None

    def _solve_system(self, b: cp.ndarray):
        """Solve Ax = b using cuDSS.

        Performs analysis, factorization, and solve phases of cuDSS.

        Parameters
        ----------
        b : cupy.ndarray
            Right-hand side vector or matrix.
            Shape: (n,) for vector or (n, nrhs) for multiple RHS.

        Returns
        -------
        x : cupy.ndarray
            Solution vector or matrix with same shape as b.

        Raises
        ------
        RuntimeError
            If cuDSS operations fail or matrix is not in CSR format.
        """
        if self._handle is None or self._config is None or self._data is None:
            self._compute_factorization()

        # Ensure b is 2D for cuDSS (n, nrhs)
        is_1d = b.ndim == 1
        if is_1d:
            b = b.reshape(-1, 1)

        nrows, ncols = self._matrix._data.shape
        nrhs = b.shape[1]

        # Create solution array
        x = cp.zeros_like(b)

        # pylint: disable=protected-access
        # Extract CSR components from matrix
        csrA_data = self._matrix._data.data
        csrA_indices = self._matrix._data.indices
        csrA_indptr = self._matrix._data.indptr

        # Create matrix objects for cuDSS
        # Matrix A (sparse, CSR format)
        matA = nm.matrix_create_csr(
            nrows,
            ncols,
            self._matrix._data.nnz,
            csrA_indptr.data.ptr,
            csrA_indices.data.ptr,
            csrA_data.data.ptr,
            nm.IndexBase.ZERO,
            csrA_data.dtype,
        )

        # Matrix b (dense, column-major)
        matb = nm.matrix_create_dn(
            nrows,
            nrhs,
            b.data.ptr,
            nrows,  # leading dimension
            b.dtype,
        )

        # Matrix x (dense, column-major)
        matx = nm.matrix_create_dn(
            nrows,
            nrhs,
            x.data.ptr,
            nrows,  # leading dimension
            x.dtype,
        )

        try:
            # Execute analysis phase (reordering and symbolic factorization)
            nm.execute(self._handle, nm.Phase.ANALYSIS, self._config, self._data, matA, matx, matb)

            # Execute factorization phase (numerical factorization)
            nm.execute(
                self._handle, nm.Phase.FACTORIZATION, self._config, self._data, matA, matx, matb
            )

            # Execute solve phase
            nm.execute(self._handle, nm.Phase.SOLVE, self._config, self._data, matA, matx, matb)

        finally:
            # Clean up matrix objects
            nm.matrix_destroy(matA)
            nm.matrix_destroy(matb)
            nm.matrix_destroy(matx)

        # Return to original shape if input was 1D
        if is_1d:
            x = x.ravel()

        return x