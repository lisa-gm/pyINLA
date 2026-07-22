# src/dalia/backend/linalg/solvers/sparse/cudss.py
from .sparse_solver import SparseSolver
from dalia.backend.config import cupy_version, nvmath_version
import numpy as np

if cupy_version is not None:
    import cupy as cp
if nvmath_version is not None:
    import nvmath as nm
    from nvmath.bindings import cudss

class CuDSS(SparseSolver):
    """CUDA Sparse Solver (cuDSS) for solving sparse linear systems.

    Uses NVIDIA's cuDSS library for GPU-accelerated sparse direct solving.
    Supports sparse matrices stored in CSR format on GPU memory.
    """

    # 1. Class attributes (if any)
    # 2. Initialization

    def __init__(self, matrix, rhs=1):
        if cupy_version is None:
            raise ImportError("CuPy is required for CuDSS. Please install CuPy to use this solver.")
        if nvmath_version is None:
            raise ImportError("NVMATH is required for CuDSS. Please install NVMATH to use this solver.")
        super().__init__(matrix)

        self._is_analyzed = False
        self._is_factorized = False

        self._data = matrix._data.data
        self._indices = matrix._data.indices
        self._indptr = matrix._data.indptr
        self._n = len(self._indptr) - 1
        self._nnz = len(self._data)
        self._rhs = rhs
        self._dtype = matrix._data.dtype
        if self._dtype == np.float32:
            self._cudss_dtype = nm.CudaDataType.CUDA_R_32F
        elif self._dtype == np.float64:
            self._cudss_dtype = nm.CudaDataType.CUDA_R_64F 
        elif self._dtype == np.complex64:
            self._cudss_dtype = nm.CudaDataType.CUDA_C_32F
        elif self._dtype == np.complex128:
            self._cudss_dtype = nm.CudaDataType.CUDA_C_64F
        

        self._d_data = cp.asarray(self._data, dtype=self._dtype)
        self._d_indices = cp.asarray(self._indices, dtype=cp.int32)
        self._d_indptr = cp.asarray(self._indptr, dtype=cp.int32)

        self._A = cudss.matrix_create_csr(
            self._n,  # nrows
            self._n,  # ncols
            self._nnz,  # nnz
            self._d_indptr.data.ptr,  # row_start (beginning of row offset array)
            0,  # row_end (NULL/0 - not used in standard CSR)
            self._d_indices.data.ptr,  # column indices
            self._d_data.data.ptr,  # values
            nm.CudaDataType.CUDA_R_32I,  # index type (int32)
            self._cudss_dtype,  # value type (complex128)
            cudss.MatrixType.GENERAL,  # matrix type (general)
            cudss.MatrixViewType.FULL,  # matrix view (full)
            cudss.IndexBase.ZERO,  # 0-based indexing
        )

        # Create right-hand side and solution vectors of the given batchsize
        # The remainder of n % batchsize will be handled by padding with zeros
        self._b = cp.zeros((self._n, self._rhs), dtype=self._dtype)

        self._b = cudss.matrix_create_dn(
            self._n,  # nrows
            self._rhs,  # ncols (number of RHS)
            self._n,  # leading dimension
            self._b.data.ptr,  # values
            self._cudss_dtype,  # complex128
            cudss.Layout.COL_MAJOR,  # column-major (Fortran style)
        )

        self._x = cp.zeros((self._n, self._rhs), dtype=self._dtype)
        self._x = cudss.matrix_create_dn(
            self._n,  # nrows
            self._rhs,  # ncols (number of RHS)
            self._n,  # leading dimension
            self._x.data.ptr,  # values
            self._cudss_dtype,  # complex128
            cudss.Layout.COL_MAJOR,  # column-major (Fortran style)
        )

        # Create cuDSS handle
        self.cudss_handle = cudss.create()
        """
        self.cudss_handle = cudss.create_mg(
            device_count=N_GPUS[0], device_indices=DEVICE_INDICES
        )
        """
        self.cudss_config = cudss.config_create()
        """
        cudss.config_set(
            self.cudss_config,
            param=cudss.ConfigParam.DEVICE_COUNT,
            value=N_GPUS.ctypes.data,
            size_in_bytes=N_GPUS.nbytes,
        )
        cudss.config_set(
            self.cudss_config,
            param=cudss.ConfigParam.DEVICE_INDICES,
            value=DEVICE_INDICES.ctypes.data,
            size_in_bytes=DEVICE_INDICES.nbytes,
        )
        """
        self.cudss_data = cudss.data_create(self.cudss_handle)




    # 3. Special representation methods
    # 4. Properties (grouped together)
    # 5. Comparison operators (if needed)
    # 6. Arithmetic operators (standard order)
    # 7. Right-hand operators (same order as above)
    # 8. In-place operators (if supported)
    # 9. Other special methods
    # 10. Public methods
    def analyze(self):
        self._analyze()
    # 11. Private/protected methods (start with _)

    def _analyze(self):
        cudss.execute(
            self.cudss_handle,
            cudss.Phase.ANALYSIS,
            self.cudss_config,
            self.cudss_data,
            self._A,
            self._x,
            self._b,
        )
        self._is_analyzed = True

    def _compute_factorization(self, overwrite: bool = False):
        
        if not self._is_analyzed:
            self._analyze()

        cudss.execute(
            self.cudss_handle,
            cudss.Phase.FACTORIZATION,
            self.cudss_config,
            self.cudss_data,
            self._A,
            self._x,
            self._b,
        )

        self._is_factorized = True
        return None # cuDSS does not support separate factorization step, so we return None

    def _solve_system(self, b):

        if not self._is_factorized:
            self._compute_factorization()

        if isinstance(b, np.ndarray):
            b = cp.asarray(b)

        cudss.matrix_destroy(self._b)
        cudss.matrix_destroy(self._x)
        self._b = b
        self._rhs = b.shape[1] if b.ndim > 1 else 1

        self._b = cudss.matrix_create_dn(
            self._n,  # nrows
            self._rhs,  # ncols (number of RHS)
            self._n,  # leading dimension
            self._b.data.ptr,  # values
            self._cudss_dtype,  # complex128
            cudss.Layout.COL_MAJOR,  # column-major (Fortran style)
        )

        x = cp.zeros((self._n, self._rhs), dtype=self._dtype)
        self._x = cudss.matrix_create_dn(
            self._n,  # nrows
            self._rhs,  # ncols (number of RHS)
            self._n,  # leading dimension
            x.data.ptr,  # values
            self._cudss_dtype,  # complex128
            cudss.Layout.COL_MAJOR,  # column-major (Fortran style)
        )

        cudss.execute(
            self.cudss_handle,
            cudss.Phase.SOLVE,
            self.cudss_config,
            self.cudss_data,
            self._A,
            self._x,
            self._b,
        )

        return x if self._rhs > 1 else x.ravel()
    