Title: cuSOLVERMp C API — cuSOLVERMp

URL Source: https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html

Published Time: Tue, 12 Aug 2025 19:10:22 GMT

Markdown Content:
cuSOLVERMp C API[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermp-c-api "Link to this heading")
-------------------------------------------------------------------------------------------------------------------------

Library Management[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#library-management "Link to this heading")
-----------------------------------------------------------------------------------------------------------------------------

### `cusolverMpCreate`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpcreate "Link to this heading")

cusolverStatus_t cusolverMpCreate(
 cusolverMpHandle_t *handle,
 int device,
 cudaStream_t stream)

This function initializes the cuSOLVERMp library handle ([cusolverMpHandle_t](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermphandle-t-label)) which holds the cuSOLVERMp library context. It allocates light hardware resources on the host, and must be called prior to making any other cuSOLVERMp library calls.

Calling any cuSOLVERMp function which uses [cusolverMpHandle_t](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermphandle-t-label) without a previous call of [cusolverMpCreate()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpcreate-label) will return an error.

The cuSOLVERMp library context is tied to the CUDA device provided by `device` and the CUDA stream `stream`.

Only one handle per process and per GPU supported. Sharing a device with multiple processes will result in undefined behavior. Currently, `stream` cannot be the default (`NULL` or `0`) stream.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | Out | cuSOLVERMp library handle. |
| device | Host | In | Device that will be assigned to the handle. |
| stream | Host | In | Stream that will be assigned to the handle. |

### `cusolverMpDestroy`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpdestroy "Link to this heading")

cusolverStatus_t cusolverMpDestroy(
 cusolverMpHandle_t handle)

This function destroys the cuSOLVERMp library handle ([cusolverMpHandle_t](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermphandle-t-label)) which holds the cuSOLVERMp library context.

The cuSOLVERMp library context is tied to the CUDA device provided by `device`. Only one handle per process and per GPU supported.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In/Out | cuSOLVERMp library handle. |

### `cusolverMpGetStream`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetstream "Link to this heading")

cusolverStatus_t cusolverMpGetStream(
 cusolverMpHandle_t handle,
 cudaStream_t *stream)

This function returns the `stream` associated to the handle.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| stream | Host | Out | Stream associated with the handle. |

* * *

### `cusolverMpGetVersion`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetversion "Link to this heading")

cusolverStatus_t cusolverMpGetVersion(
 cusolverMpHandle_t handle,
 int *version)

This function returns the version number of the cuSOLVERMp library.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| version | Host | Out | cuSOLVERMp library version. Value is `CUSOLVERMP_VER_MAJOR * 1000 + CUSOLVERMP_VER_MINOR * 100 + CUSOLVERMP_VER_PATCH`. |

* * *

### `cusolverMpSetMathMode`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsetmathmode "Link to this heading")

cusolverStatus_t cusolverMpSetMathMode(
 cusolverMpHandle_t handle,
 cusolverMathMode_t mode)

This function sets the math mode for the cuSOLVERMp library handle. It will be propagated to the internal cuBLAS and cuSOLVER handles. The math mode controls the behavior of mathematical operations performed by the library.

Note

Please note that the workspace sizes returned by `*_bufferSize` APIs may depend on the math mode.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| mode | Host | In | Math mode to be set for the handle. Available options are `CUSOLVER_DEFAULT_MATH` and `CUSOLVER_FP32_EMULATED_BF16X9_MATH`. See [cusolverMathMode_t](https://docs.nvidia.com/cuda/cusolver/index.html.md#cusolvermathmode-t) for details. |

* * *

### `cusolverMpGetMathMode`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetmathmode "Link to this heading")

cusolverStatus_t cusolverMpGetMathMode(
 cusolverMpHandle_t handle,
 cusolverMathMode_t *mode)

This function retrieves the current math mode from the cuSOLVERMp library handle.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| mode | Host | Out | Current math mode of the handle. |

* * *

### `cusolverMpSetEmulationStrategy`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsetemulationstrategy "Link to this heading")

cusolverStatus_t cusolverMpSetEmulationStrategy(
 cusolverMpHandle_t handle,
 cudaEmulationStrategy_t strategy)

This function sets the emulation strategy for the cuSOLVERMp library handle. It will be propagated to the internal cuBLAS and cuSOLVER handles.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| strategy | Host | In | Emulation strategy to be set for the handle. Available options are `CUDA_EMULATION_STRATEGY_DEFAULT`, `CUDA_EMULATION_STRATEGY_PERFORMANT`, and `CUDA_EMULATION_STRATEGY_EAGER`. For more information about the effects of the corresponding strategies, please refer to the analogous definition of [cublasEmulationStrategy_t](https://docs.nvidia.com/cuda/cublas/index.html.md#cublasemulationstrategy-t). |

The emulation strategy only has an effect if the math mode is set to `CUSOLVER_FP32_EMULATED_BF16X9_MATH`.

* * *

### `cusolverMpGetEmulationStrategy`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetemulationstrategy "Link to this heading")

cusolverStatus_t cusolverMpGetEmulationStrategy(
 cusolverMpHandle_t handle,
 cudaEmulationStrategy_t *strategy)

This function retrieves the current emulation strategy from the cuSOLVERMp library handle.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| strategy | Host | Out | Current emulation strategy of the handle. |

* * *

Grid Management[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#grid-management "Link to this heading")
-----------------------------------------------------------------------------------------------------------------------

### `cusolverMpCreateDeviceGrid`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpcreatedevicegrid "Link to this heading")

cusolverStatus_t cusolverMpCreateDeviceGrid(
 cusolverMpHandle_t handle,
 cusolverMpGrid_t *grid,
 ncclComm_t comm,
 int32_t numRowDevices,
 int32_t numColDevices,
 cusolverMpGridMapping_t mapping)

This function initializes the [grid](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermpgrid-t-label) opaque data structure. It maps the given resources (communicator, grid dimensions and grid layout) to a grid object.

All the processes defined to be in this grid must enter this function.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| grid | Host | Out | Grid object to be initialized. |
| comm | Host | In | Communicator that will be associated with the grid. |
| numRowDevices | Host | In | How many process rows the grid will contain. |
| numColDevices | Host | In | How many process columns the grid will contain. |
| mapping | Host | In | How to map processes to the grid. See description of [cusolverMpGrid_t](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermpgrid-t-label) for further details. Currently, only `CUSOLVERMP_GRID_MAPPING_COL_MAJOR` is supported. |

* * *

### `cusolverMpDestroyGrid`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpdestroygrid "Link to this heading")

cusolverStatus_t cusolverMpDestroyGrid(
 cusolverMpGrid_t grid)

This function destroys the given `grid` object.

All the processes defined to be in this grid must enter this function.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| grid | Host | In/Out | Grid object to be destroyed. |

* * *

Matrix Management[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#matrix-management "Link to this heading")
---------------------------------------------------------------------------------------------------------------------------

### `cusolverMpCreateMatrixDesc`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpcreatematrixdesc "Link to this heading")

cusolverStatus_t cusolverMpCreateMatrixDesc(
 cusolverMpMatrixDescriptor_t *descr,
 cusolverMpGrid_t grid,
 cudaDataType dataType,
 int64_t M_A,
 int64_t N_A,
 int64_t MB_A,
 int64_t NB_A,
 uint32_t RSRC_A,
 uint32_t CSRC_A,
 int64_t LLD_A)

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| descr | Host | Out | Matrix descriptor object initialized by this function. |
| dataType | Host | In | Data type of the matrix A. |
| M_A | Host | In | Number of rows in the global matrix A. |
| N_A | Host | In | Number of columns in the global matrix A. |
| MB_A | Host | In | Blocking factor used to distribute the rows of the global matrix A. |
| NB_A | Host | In | Blocking factor used to distribute the columns of the global matrix A. |
| RSRC_A | Host | In | Process row over which the first row of the matrix A is distributed. Only the value of `0` is currently supported. |
| CSRC_A | Host | In | Process column over which the first row of the matrix A is distributed. Only the value of `0` is currently supported. |
| LLD_A | Host | In | Leading dimension of the local matrix. |

Supported values for `dataType` argument are listed below:

| Data Type of A | Description |
| --- | --- |
| CUDA_R_32I | 32-bit integer values. |
| CUDA_R_64I | 64-bit integer values. |
| CUDA_R_32F | Single precision real values. |
| CUDA_R_64F | Double precision real values. |
| CUDA_C_32F | Single precision complex values. |
| CUDA_C_64F | Double precision complex values. |

* * *

### `cusolverMpDestroyMatrixDesc`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpdestroymatrixdesc "Link to this heading")

cusolverStatus_t cusolverMpDestroyMatrixDesc(
 cusolverMpMatrixDescriptor_t descr )

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| descr | Host | In/Out | Matrix descriptor object destroyed by this function. |

* * *

Utility[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#utility "Link to this heading")
-------------------------------------------------------------------------------------------------------

### `cusolverMpNUMROC`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpnumroc "Link to this heading")

int64_t cusolverMpNUMROC(
 int64_t n,
 int64_t nb,
 uint32_t iproc,
 uint32_t isrcproc,
 uint32_t nprocs)

Computes the number of rows or columns of a distributed matrix owned by the process indicated by `iproc` argument.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| n | Host | In | Number of rows or columns in the global distributed matrix. |
| nb | Host | In | Row or column blocking size of the global matrix. |
| iproc | Host | In | The coordinate of the process whose local array row or column is to be determined. |
| isrcproc | Host | In | The coordinate of the process that owns the first row or column of the distributed matrix. |
| nprocs | Host | In | The total number of row or column processes over which the matrix is distributed. |

Returns the number of rows or columns of a distributed matrix owned by the process indicated by `iproc` argument.

* * *

### `cusolverMpMatrixGatherD2H`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpmatrixgatherd2h "Link to this heading")

cusolverStatus_t cusolverMpMatrixGatherD2H(
 cusolverMpHandle_t handle,
 int64_t M,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 int root,
 void *h_dst,
 int64_t h_lddst)

Gathers the global distributed matrix `A` on a buffer provided on process `root`. The input matrix `A` is originally distributed using 2D block-cyclic format, on output `h_dst` contains the matrix in column-major format.

Notice that, for this function, the input data is on the device and the output is stored on host memory.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| M | Host | In | Number of rows of the global distributed matrix A. |
| N | Host | In | Number of columns of the global distributed matrix A. |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. On entry, this array contains the local pieces of the M-by-N distributed matrix sub(A). |
| IA | Host | In | Row index in the global matrix A indicating the first row of sub(A). This function does not make any assumptions on the alignment of `IA`. |
| JA | Host | In | Column index in the global matrix A indicating the first column of sub(A). This function does not make any assumptions on the alignment of `JA`. |
| descrA | Host | In | Matrix descriptor of the global matrix A. |
| root | Host | In | Process ID on which the matrix A will be gathered. |
| h_dst | Host | Out | Destination host buffer on `root` process. On output it contains the global matrix A stored in column-major format. Total size must be at least `M*N` words. |
| h_lddst | Host | In | Leading dimension of the `h_dst` on `root` process. Must be larger than `M`. |

Return Status:

Warning

This function is meant as a utility function to verify correctness of the data layouts and it is not intended to achieve high performance on large inputs.

* * *

### `cusolverMpMatrixScatterH2D`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpmatrixscatterh2d "Link to this heading")

cusolverStatus_t cusolverMpMatrixScatterH2D(
 cusolverMpHandle_t handle,
 int64_t M,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 int root,
 void *h_src,
 int64_t h_ldsrc)

Scatters the matrix stored in the local buffer `h_src` from `root` process to a distributed global matrix `A`.

The input matrix `h_src` is stored in column-major format. On output, `d_A` contains the local portions of the global matrix `A` distributed in 2D block-cyclic format.

Notice that, for this function, the input data is on the host and the output is stored on device memory.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| M | Host | In | Number of rows of the global distributed matrix A. |
| N | Host | In | Number of columns of the global distributed matrix A. |
| d_A | Device | Out | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. On output, this array contains the local pieces of the M-by-N distributed matrix sub(A). |
| IA | Host | In | Row index in the global matrix A indicating the first row of sub(A). This function does not make any assumptions on the alignment of `IA`. |
| JA | Host | In | Column index in the global matrix A indicating the first column of sub(A). This function does not make any assumptions on the alignment of `JA`. |
| descrA | Host | In | Matrix descriptor of the global matrix A. |
| root | Host | In | Process ID which the matrix A will be scattered from. |
| h_src | Host | In | Source buffer on `root` process. On input it contains the global `M` by `N` matrix A stored in column-major format. |
| h_ldsrc | Host | In | Leading dimension of the `h_src` on `root` process. Must be larger than `M`. |

Warning

This function is meant as a utility function to verify correctness of the data layouts and it is not intended to achieve high performance on large inputs.

* * *

Logging[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#logging "Link to this heading")
-------------------------------------------------------------------------------------------------------

### `cusolverMpLoggerSetCallback`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermploggersetcallback "Link to this heading")

cusolverStatus_t cusolverMpLoggerSetCallback(
 cusolverMpLoggerCallback_t callback)

This function sets the logging callback function.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| callback | Host | In | Pointer to a callback function. See [cusolverMpLoggerCallback_t](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermploggercallback-t-label). |

Warning

This is an experimental feature.

* * *

### `cusolverMpLoggerSetFile`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermploggersetfile "Link to this heading")

cusolverStatus_t cusolverMpLoggerSetFile(
 FILE *file)

This function sets the logging output file. Note: once registered using this function call, the provided file handle must not be closed unless the function is called again to switch to a different file handle.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| file | Host | In | Pointer to an open file. File should have write permission. |

Warning

This is an experimental feature.

* * *

### `cusolverMpLoggerOpenFile`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermploggeropenfile "Link to this heading")

cusolverStatus_t cusolverMpLoggerOpenFile(
 const char* logFile)

This function opens a logging output file in the given path.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| logFile | Host | In | Path of the logging output file. |

Warning

This is an experimental feature.

* * *

### `cusolverMpLoggerSetLevel`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermploggersetlevel "Link to this heading")

cusolverStatus_t cusolverMpLoggerSetLevel(
 int level)

This function sets the logging level for cuSOLVERMp library. The logging level controls the verbosity of the logging output.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| level | Host | In | Value of the logging level. See `cuSOLVERMp Logging`. |

Warning

This is an experimental feature.

* * *

### `cusolverMpLoggerSetMask`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermploggersetmask "Link to this heading")

cusolverStatus_t cusolverMpLoggerSetMask(
 int mask)

This function sets the value of the logging mask.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| mask | Host | In | Value of the logging mask. See `cuSOLVERMp Logging`. |

Warning

This is an experimental feature.

* * *

### `cusolverMpLoggerForceDisable`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermploggerforcedisable "Link to this heading")

cusolverStatus_t cusolverMpLoggerForceDisable(
 int level)

This function disables logging for the entire run.

Warning

This is an experimental feature.

* * *

Dense Linear Algebra APIs[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#dense-linear-algebra-apis "Link to this heading")
-------------------------------------------------------------------------------------------------------------------------------------------

### `cusolverMpGetrf`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf "Link to this heading")

cusolverStatus_t cusolverMpGetrf(
 cusolverMpHandle_t handle,
 int64_t M,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 int64_t *d_ipiv,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

This routine computes an LU factorization of a general M-by-N distributed matrix sub(A) using partial pivoting. The user can also disable pivoting by setting `d_ipiv=NULL`.

The factorization has the form:

where  is a permutation matrix,  is lower triangular with unit diagonal elements (lower trapezoidal if ), and  is upper triangular (upper trapezoidal if ).  and  are stored in sub(A).

The user can combine [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label) and [cusolverMpGetrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs-label) to solve a system of linear equations.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| d_A | Device | In/Out | Pointer to the first entry of the local portion of the global matrix A. On output, the sub(A) is overwritten with the L and U factors. |
| IA | Host | In | Row index of the first row of the sub(A). This function does not make any assumptions on the alignment of `IA`. |
| JA | Host | In | Column index of the first column of the sub(A). This function does not make any assumptions on the alignment of `JA`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_ipiv | Device | Out | Local array of dimension `(LOCr(M_A)+MB_A)`. If the user set `d_ipiv != NULL`, on output, this array contains the pivoting information. `d_ipiv[i]` indicates the global row local row i was swapped with. This array is tied to the distributed matrix A. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpGetrf_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpGetrf_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. `info > 0` indicates the index of the leading minor in the case of a singular matrix. |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpGetrf_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-buffersize "Link to this heading")

cusolverStatus_t cusolverMpGetrf_bufferSize(
 cusolverMpHandle_t handle,
 int64_t M,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 int64_t *d_ipiv,
 cudaDataType_t computeType,
 size_t *workspaceInBytesOnDevice,
 size_t *workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label).

The user can set `d_ipiv=NULL` so [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label) will compute the LU factorization of the input matrix A without pivoting.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| d_A | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| IA | Host | In | Row index of the first row of the sub(A). This function does not make any assumptions on the alignment of `IA`. |
| JA | Host | In | Column index of the first column of the sub(A). This function does not make any assumptions on the alignment of `JA`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_ipiv | Device | In | Indicates a pointer to a distributed integer array. When it is not `null`, workspace for pivoting is accounted. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label). |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpGetrs`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs "Link to this heading")

cusolverStatus_t cusolverMpGetrs(
 cusolverMpHandle_t handle,
 cublasOperation_t trans,
 int64_t N,
 int64_t NRHS,
 const void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 const int64_t *d_ipiv,
 void *d_B,
 int64_t IB,
 int64_t JB,
 cusolverMpMatrixDescriptor_t descrB,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *d_info)

This routine solves a system of distributed linear equations

with a general N-by-N distributed matrix sub(A) using the LU factorization computed by [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label).

Where  is defined by the argument `trans`, which allows to solve linear systems of the form:

| trans | Form of the linear system |
| --- | --- |
| CUBLAS_OP_N |  |
| CUBLAS_OP_T |  |
| CUBLAS_OP_C |  |

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| trans | Host | In | Specifies the form of the linear system. Only `CUBLAS_OP_N` is currently supported. |
| N | Host | In | Number of rows of sub(A). |
| NRHS | Host | In | Number of columns of sub(B). Currently, this routine only supports `NRHS=1`. |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. On entry, this array contains the local pieces of the M-by-N distributed L and U factors of sub(A) as computed by [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label). |
| IA | Host | In | Row index of the first row of the sub(A). This function does not make any assumptions on the alignment of `IA`. |
| JA | Host | In | Column index of the first column of the sub(A). This function does not make any assumptions on the alignment of `JA`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_ipiv | Device | In | Local array of dimension `(LOCr(M_A)+MB_A)` containing the pivoting information as computed by [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label). |
| d_B | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_B,LOCc(JB+NRHS-1))`. On entry, the right hand sides sub(B). On exit, sub(B) is overwritten by the solution distributed matrix X. |
| IB | Host | In | Row index of the first row of the sub(B). This function does not make any assumptions on the alignment of `IB`. |
| JB | Host | In | Column index of the first column of the sub(B). This function does not make any assumptions on the alignment of `JB`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpGetrs_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpGetrs_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. `info > 0` indicates the index of the leading minor in the case of a singular matrix. |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpGetrs_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs-buffersize "Link to this heading")

cusolverStatus_t cusolverMpGetrs_bufferSize(
 cusolverMpHandle_t handle,
 cublasOperation_t trans,
 int64_t N,
 int64_t NRHS,
 const void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 const int64_t *d_ipiv,
 void *d_B,
 int64_t IB,
 int64_t JB,
 cusolverMpMatrixDescriptor_t descrB,
 cudaDataType_t computeType,
 size_t *workspaceInBytesOnDevice,
 size_t *workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpGetrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs-label).

If pivoting was disabled during [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label), the user must set `d_ipiv=NULL`.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| trans | Host | In | Specifies the form of the linear system. Only `CUBLAS_OP_N` is currently supported. |
| N | Host | In | Number of rows of sub(A). |
| NRHS | Host | In | Number of columns of sub(B). Currently, this routine only supports `NRHS=1`. |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. On entry, this array contains the local pieces of the M-by-N distributed L and U factors of sub(A) as computed by [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label). |
| IA | Host | In | Row index of the first row of the sub(A). This function does not make any assumptions on the alignment of `IA`. |
| JA | Host | In | Column index of the first column of the sub(A). This function does not make any assumptions on the alignment of `JA`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_ipiv | Device | In | Local array of dimension `(LOCr(M_A)+MB_A)` containing the pivoting information as computed by [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label). |
| d_B | Device | In | Pointer to the first entry of the local portion of the global matrix B. On output, B is overwritten the solution of the linear system. |
| IB | Host | In | Row index of the first row of the sub(B). This function does not make any assumptions on the alignment of `IB`. |
| JB | Host | In | Column index of the first column of the sub(B). This function does not make any assumptions on the alignment of `JB`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cusolverMpGetrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cusolverMpGetrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs-label). |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpPotrf`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf "Link to this heading")

cusolverStatus_t cusolverMpPotrf(
 cusolverMpHandle_t handle,
 cublasFillMode_t uplo,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Computes the Cholesky factorization of an N-by-N real symmetric or a complex hermitian positive definite distributed matrix sub(A) denoting `A(IA:IA+N-1, JA:JA+N-1)`.

If A is upper triangular and `uplo=CUBLAS_FILL_MODE_UPPER`, the factorization has the form

where U is upper triangular.

If the matrix is lower triangular and `uplo` is set to `CUBLAS_FILL_MODE_LOWER`, the factorization has the form

where L is lower triangular.

The user can combine [cusolverMpPotrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-label) and [cusolverMpPotrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs-label) to solve a system of linear equations.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| uplo | Host | In | Specifies if A is upper (`CUBLAS_FILL_MODE_UPPER`) or lower triangular matrix (`CUBLAS_FILL_MODE_LOWER`). |
| N | Host | In | Number of rows and columns of sub(A). |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. On entry, this array contains the local pieces of the N-by-N distributed matrix sub(A). On output, this array contains the L or U factors of A, depending on the value of `uplo`. |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A).`JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpPotrf_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpPotrf_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. `info > 0` indicates the index of the leading minor in the case of a singular matrix. |

This function requires square block size `(MB_A == NB_A)`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpPotrf_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-buffersize "Link to this heading")

cusolverStatus_t cusolverMpPotrf_bufferSize(
 cusolverMpHandle_t handle,
 cublasFillMode_t uplo,
 int64_t N,
 const void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 cudaDataType_t computeType,
 size_t* workspaceInBytesOnDevice,
 size_t* workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpPotrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| uplo | Host | In | Specifies if A is upper (`CUBLAS_FILL_MODE_UPPER`) or lower triangular matrix (`CUBLAS_FILL_MODE_LOWER`). |
| N | Host | In | Number of rows and columns of sub(A). |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. On entry, this array contains the local pieces of the N-by-N distributed matrix sub(A). On output, this array contains the L or U factors of A, depending on the value of `uplo`. |
| IA | Host | In | Row index of the first row of the sub(A). This function does not make any assumptions on the alignment of `IA`. |
| JA | Host | In | Column index of the first column of the sub(A). This function does not make any assumptions on the alignment of `JA`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cusolverMpPotrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cusolverMpPotrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-label). |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpPotrs`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs "Link to this heading")

cusolverStatus_t cusolverMpPotrs(
 cusolverMpHandle_t handle,
 cublasFillMode_t uplo,
 int64_t N,
 int64_t NRHS,
 const void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 void *d_B,
 int64_t IB,
 int64_t JB,
 cusolverMpMatrixDescriptor_t descrB,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Solves a system of linear equations

where sub(A) denotes `A(IA:IA+N-1,JA:JA+N-1)` and is a N-by-N symmetric or hermitian positive definite distributed matrix using the Cholesky factorization:

or

computed by [cusolverMpPotrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-label) and sub(B) denotes the distributed matrix `B(IB:IB+N-1,JB:JB+NRHS-1)`.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| uplo | Host | In | Specifies if A is upper (`CUBLAS_FILL_MODE_UPPER`) or lower triangular matrix (`CUBLAS_FILL_MODE_LOWER`). |
| N | Host | In | Number of rows and columns of sub(A). |
| NRHS | Host | In | Number of columns of sub(B). Currently, this routine only supports `NRHS=1`. |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. Contains the local pieces of the N-by-N distributed L or U factors of sub(A) as computed by [cusolverMpPotrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-label). |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `NB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `MB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_B | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_B,LOCc(JB+NRHS-1))`. On entry, the right hand sides sub(B). On exit, sub(B) is overwritten by the solution distributed matrix X. |
| IB | Host | In | Row index of the first row of the sub(B). This function does not make any assumptions on the alignment of `IB`. |
| JB | Host | In | Column index of the first column of the sub(B). This function does not make any assumptions on the alignment of `JB`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpPotrs_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpPotrs_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. `info > 0` indicates the index of the leading minor in the case of a singular matrix. |

This function requires square block size `(MB_A == NB_A)` and alignment of sub(A) and sub(B) matrices, meaning `(MB_A == MB_B)` and `(IA == IB)`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

—

#### `cusolverMpPotrs_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs-buffersize "Link to this heading")

cusolverStatus_t cusolverMpPotrs_bufferSize(
 cusolverMpHandle_t handle,
 cublasFillMode_t uplo,
 int64_t n,
 int64_t nrhs,
 const void *a,
 int64_t ia,
 int64_t ja,
 cusolverMpMatrixDescriptor_t descrA,
 const void *b,
 int64_t ib,
 int64_t jb,
 cusolverMpMatrixDescriptor_t descB,
 cudaDataType_t computeType,
 size_t* workspaceInBytesOnDevice,
 size_t* workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpPotrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| uplo | Host | In | Specifies if A is upper (`CUBLAS_FILL_MODE_UPPER`) or lower triangular matrix (`CUBLAS_FILL_MODE_LOWER`). |
| N | Host | In | Number of rows and columns of sub(A). |
| NRHS | Host | In | Number of columns of sub(B). Currently, this routine only supports `NRHS=1`. |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. Contains the local pieces of the N-by-N distributed L or U factors of sub(A) as computed by [cusolverMpPotrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-label). |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `NB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `MB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_B | Device | In | Pointer into the local memory to an array of dimension `(LLD_B,LOCc(JB+NRHS-1))`. On entry, the right hand sides sub(B). On exit, sub(B) is overwritten by the solution distributed matrix X. |
| IB | Host | In | Row index of the first row of the sub(B). This function does not make any assumptions on the alignment of `IB`. |
| JB | Host | In | Column index of the first column of the sub(B). This function does not make any assumptions on the alignment of `JB`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cusolverMpPotrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cusolverMpPotrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs-label). |

This function requires square block size `(MB_A == NB_A)` and alignment of sub(A) and sub(B) matrices, meaning `(MB_A == MB_B)` and `(IA == IB)`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

—

### `cusolverMpGeqrf`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf "Link to this heading")

cusolverStatus_t cusolverMpGeqrf(
 cusolverMpHandle_t handle,
 int64_t M,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 void *d_tau,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Computes the QR factorization of a distributed M-by-N matrix sub(A) denoting `A(IA:IA+M-1, JA:JA+N-1)`.

where Q is an orthogonal matrix represented by a product of Householder reflectors with the array of `tau` and R is upper triangular matrix.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| d_A | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. On entry, this array contains the local pieces of the M-by-N distributed matrix sub(A). On output, this array contains the R factors of A and Householder reflectors below of diagonals with `tau` vector. |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_tau | Device | Out | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. This array contains scalar factors of the Householder reflectors |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpGeqrf_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpGeqrf_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

This function requires square block size `(MB_A == NB_A)`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpGeqrf_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-buffersize "Link to this heading")

cusolverStatus_t cusolverMpGeqrf_bufferSize(
 cusolverMpHandle_t handle,
 int64_t M,
 int64_t N,
 const void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 cudaDataType_t computeType,
 size_t* workspaceInBytesOnDevice,
 size_t* workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpGeqrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. |
| IA | Host | In | Row index in the global matrix A indicating the first row of sub(A). This function does not make any assumptions on the alignment of `IA`. |
| JA | Host | In | Column index in the global matrix A indicating the first column of sub(A). This function does not make any assumptions on the alignment of `JA`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cusolverMpGeqrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cusolverMpGeqrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-label). |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

—

### `cusolverMpOrmqr`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormqr "Link to this heading")

cusolverStatus_t cusolverMpOrmqr(
 cusolverMpHandle_t handle,
 cublasSideMode_t side,
 cublasOperation_t trans,
 int64_t M,
 int64_t N,
 int64_t K,
 const void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 const void *d_tau,
 void *d_C,
 int64_t IC,
 int64_t JC,
 const cusolverMpMatrixDescriptor_t descrC,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Multiply distributed M-by-N matrix sub(C) denoting `C(IC:IC+M-1, JC:JC+N-1)` by the orthogonal matrix Q can be given from [cusolverMpGeqrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-label).

The function can perform the following matrix product and overwrite the result on sub(C).

for the `side` of `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT` respectively. Note that the current implementation only support for `CUBLAS_SIDE_LEFT`.

Q is a orthogonal matrix formed as the product of Householder reflectors returned from [cusolverMpGeqrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-label).

The number of the Householder reflectors is constrained by `K <= M` and `K <= N` for `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT` respectively.

The `op` can be translated to , ,  based on the `trans` argument `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_H`.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| side | Host | In | Indicate that Q is applied from left or right side. |
| trans | Host | In | Indicate that Q is applied with no-transpose or (conj)transpose. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| K | Host | In | Number of Householder reflectors defining Q. |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. This array contains Householder reflectors below of diagonals with `tau` vector |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A).`JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_tau | Device | In | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. This array contains scalar factors of the Householder reflectors |
| d_C | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_C, LOCc(JC+N-1))`. On entry, the array contains the local pieces of the M-by-N distributed matrix sub(C). On exit, the sub(C) is overwritten by op(Q)*sub(C) or sub(C)*op(Q). |
| IC | Host | In | Row index of the first row of the sub(C). `IC` must be a multiple of the row blocking dimension `MB_C`. |
| JC | Host | In | Column index of the first column of the sub(C). `JC` must be a multiple of the column blocking dimension `NB_C`. |
| descrC | Host | In | Matrix descriptor associated to the global matrix C. |
|  |  |  |  |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpOrmqr_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormqr-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpOrmqr_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormqr-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

This function requires square block size `(MB_A == NB_A)` and alignment of sub(A) and sub(C) matrices, meaning `(MB_A == MB_C)` and `(IA == IC)`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpOrmqr_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormqr-buffersize "Link to this heading")

cusolverStatus_t cusolverMpOrmqr_bufferSize(
 cusolverMpHandle_t handle,
 cublasSideMode_t side,
 cublasOperation_t trans,
 int64_t M,
 int64_t N,
 int64_t K,
 const void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 const void *d_tau,
 void *d_C,
 int64_t IC,
 int64_t JC,
 const cusolverMpMatrixDescriptor_t descrC,
 cudaDataType_t computeType,
 size_t* workspaceInBytesOnDevice,
 size_t* workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpOrmqr()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormqr-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| side | Host | In | Indicate that Q is applied from left or right side. |
| trans | Host | In | Indicate that Q is applied with no-transpose or (conj)transpose. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| K | Host | In | Number of Householder reflectors defining Q. |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. This array contains Householder reflectors below of diagonals with `tau` vector |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A).`JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_tau | Device | In | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. This array contains scalar factors of the Householder reflectors |
| d_C | Device | In | Pointer into the local memory to an array of dimension `(LLD_C, LOCc(JC+N-1))`. On entry, the array contains the local pieces of the M-by-N distributed matrix sub(C). On exit, the sub(C) is overwritten by op(Q)*sub(C) or sub(C)*op(Q). |
| IC | Host | In | Row index of the first row of the sub(C). `IC` must be a multiple of the row blocking dimension `MB_C`. |
| JC | Host | In | Column index of the first column of the sub(C). `JC` must be a multiple of the column blocking dimension `NB_C`. |
| descrC | Host | In | Matrix descriptor associated to the global matrix C. |
|  |  |  |  |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | The size in bytes of the local device workspace needed by [cusolverMpOrmqr()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormqr-label). |
| workspaceInBytesOnHost | Host | Out | The size in bytes of the local host workspace needed by [cusolverMpOrmqr()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormqr-label). |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpGels`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgels "Link to this heading")

cusolverStatus_t cusolverMpGels(
 cusolverMpHandle_t handle,
 cublasOperation_t trans,
 int64_t M,
 int64_t N,
 int64_t NRHS,
 void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 void *d_B,
 int64_t IB,
 int64_t JB,
 const cusolverMpMatrixDescriptor_t descrB,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Solves overdetermined or underdetermined linear systems involving a distributed M-by-N matrix sub(A) denoting `A(IA:IA+M-1, JA:JA+N-1)` or its transpose, using QR or LQ factorization of sub(A).

Note that the solution of overdetermined systems (`M >= N`) with a no-transpose option is only supported via QR factorization [cusolverMpGeqrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-label).

where sub(B) is a distributed M-by-NRHS multi-vector denoting `B(IB:IB+M-1, JB:JB+NRHS-1)` and the solution multi-vector `X` is overwritten on the sub(B).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| trans | Host | In | Indicate that the linear system of sub(A) involves with no-transpose or (conj)transpose. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| NRHS | Host | In | Number of right hand side vectors i.e., number of columns of sub(B) and X. |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_B | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_B, LOCc(JB+NRHS-1))`. On entry, the array contains the local pieces of the M-by-NRHS distributed matrix sub(B). On exit, the sub(B) is overwritten by the solution of the solution vectors. |
| IB | Host | In | Row index of the first row of the sub(B). `IB` must be a multiple of the row blocking dimension `MB_B`. |
| JB | Host | In | Column index of the first column of the sub(B). `JB` must be a multiple of the column blocking dimension `NB_B`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpGels_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgels-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpGels_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgels-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

This function requires square block size `(MB_A == NB_A)` and alignment of sub(A) and sub(B) matrices, meaning `(MB_A == MB_B)` and `(IA == IB)`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpGels_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgels-buffersize "Link to this heading")

cusolverStatus_t cusolverMpGels_bufferSize(
 cusolverMpHandle_t handle,
 cublasOperation_t trans,
 int64_t M,
 int64_t N,
 int64_t NRHS,
 void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 void *d_B,
 int64_t IB,
 int64_t JB,
 const cusolverMpMatrixDescriptor_t descrB,
 cudaDataType_t computeType,
 size_t* workspaceInBytesOnDevice,
 size_t* workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpGels()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgels-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| trans | Host | In | Indicate that the linear system of sub(A) involves with no-transpose or (conj)transpose. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| NRHS | Host | In | Number of right hand side vectors i.e., number of columns of sub(B) and X. |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_B | Device | In | Pointer into the local memory to an array of dimension `(LLD_B, LOCc(JB+NRHS-1))`. |
| IB | Host | In | Row index of the first row of the sub(B). `IB` must be a multiple of the row blocking dimension `MB_B`. |
| JB | Host | In | Column index of the first column of the sub(B). `JB` must be a multiple of the column blocking dimension `NB_B`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
|  |  |  |  |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | The size in bytes of the local device workspace needed by [cusolverMpGels()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgels-label). |
| workspaceInBytesOnHost | Host | Out | The size in bytes of the local host workspace needed by [cusolverMpGels()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgels-label). |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpSytrd`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd "Link to this heading")

cusolverStatus_t cusolverMpSytrd(
 cusolverMpHandle_t handle,
 cublasFillMode_t uplo,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 void *d_d,
 void *d_e,
 void *d_tau,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Reduces a symmetric (or hermitian for a complex value type) distributed N-by-N matrix sub(A) denoting `A(IA:IA+N-1, JA:JA+N-1)` to a tridiagonal form.

Currently, the function is implemented for `CUBLAS_FILL_MODE_LOWER` only.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| uplo | Host | In | Indicate that the function uses either upper or lower triangular part of sub(A). |
| N | Host | In | Number of rows/columns of square matrix sub(A). |
| d_A | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. On entry, the array contains the local part of symmetric distributed matrix sub(A). On exit, if the `CUBLAS_FILL_MODE_UPPER` is set, the diagonal and first superdiagonal of the tridiagonal of sub(A) is overwritten by the corresponding tridiagonal matrix, and Householder reflectors are stored above the superdiagonal of sub(A). If `CUBLAS_FILL_MODE_LOWER` is set, the diagonal and first subdiagonal of sub(A) is overwritten by the corresponding tridiagonal matrix, and Householder reflectors are stored below the subdiagonal of sub(A). |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_d | Device | Out | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. The diagonal elements of tridiagonal matrix is stored: `d(i) = A(i,i)`. |
| d_e | Device | Out | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. The off-diagonal elements of tridiagonal matrix is stored: `e(i) = A(i,i+1)` for `CUBLAS_FILL_MODE_UPPER` and `e(i) = A(i+1,i)` for `CUBLAS_FILL_MODE_LOWER`. |
| d_tau | Device | Out | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. This array contains scalar factors of the Householder reflectors |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpSytrd_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpSytrd_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

This function requires square block size `(MB_A == NB_A)`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpSytrd_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd-buffersize "Link to this heading")

cusolverStatus_t cusolverMpSytrd_bufferSize(
 cusolverMpHandle_t handle,
 cublasFillMode_t uplo,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 void *d_d,
 void *d_e,
 void *d_tau,
 cudaDataType_t computeType,
 size_t *workspaceInBytesOnDevice,
 size_t *workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpSytrd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| uplo | Host | In | Indicate that the function uses either upper or lower triangular part of sub(A). |
| N | Host | In | Number of rows/columns of square matrix sub(A). |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_d | Device | In | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. |
| d_e | Device | In | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. |
| d_tau | Device | In | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | The size in bytes of the local device workspace needed by [cusolverMpSytrd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd-label). |
| workspaceInBytesOnHost | Host | Out | The size in bytes of the local host workspace needed by [cusolverMpSytrd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd-label). |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpStedc`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc "Link to this heading")

cusolverStatus_t cusolverMpStedc(
 cusolverMpHandle_t handle,
 char *compz,
 int64_t N,
 void *d_d,
 void *d_e,
 void *d_Q,
 int64_t IQ,
 int64_t JQ,
 const cusolverMpMatrixDescriptor_t descrQ,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Computes all eigenvalues and eigenvectors of a symmetric tridiagonal matrix using the divide and conquer algorithm.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| compz | Host | In | Option to compute eigenvalues only(`compz=N`) or both eigenvalues/vectors(`compz=I`). Currently, `I` is only implemented. |
| N | Host | In | Number of rows/columns of square matrix sub(A). |
| d_d | Device | In/Out | Pointer to an array of dimension `N`. On entry, the array contains diagonal elements of the tridiagonal matrix. On exit, the eigenvalues are stored in descending order. |
| d_e | Device | In/Out | Pointer to an array of dimension `N-1`. On entry, the array contains subdiagonal elements of the tridiagonal matrix. On exit, the content of the array is destroyed. |
| d_Q | Device | Out | Pointer into the local memory to an array of dimension `(LLD_Q, LOCc(JQ+N-1))`. On output, the array contains the local elements of orthonormal eigenvectors of the symmetric diagonal matrix. |
| IQ | Host | In | Row index of the first row of the sub(Q). `IQ` must be a multiple of the row blocking dimension `MB_Q`. |
| JQ | Host | In | Column index of the first column of the sub(A). `JQ` must be a multiple of the column blocking dimension `NB_Q`. |
| descrQ | Host | In | Matrix descriptor associated to the global matrix Q. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpStedc_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpStedc_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

This function requires a square block size for Q, `(MB_Q == NB_Q)`.

This routine supports the following combinations of data types:

| Data Type of Tridiagonal Matrix | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

Note that this function uses the divide and conquer algorithm to compute eigen pairs. Currently, the function can be failed when the blocksize is not a power of two. This is a known issue and we will address the problem later.

* * *

#### `cusolverMpStedc_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc-buffersize "Link to this heading")

cusolverStatus_t cusolverMpStedc_bufferSize(
 cusolverMpHandle_t handle,
 char *compz,
 int64_t N,
 void *d_d,
 void *d_e,
 void *d_Q,
 int64_t IQ,
 int64_t JQ,
 const cusolverMpMatrixDescriptor_t descrQ,
 cudaDataType_t computeType,
 size_t *workspaceInBytesOnDevice,
 size_t *workspaceInBytesOnHost,
 int *iwork)

Computes the size in bytes of the host and device working buffers required by [cusolverMpStedc()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| compz | Host | In | Option to compute eigenvalues only(`compz=N`) or both eigenvalues/vectors(`compz=I`). Currently, `I` is only implemented. |
| N | Host | In | Number of rows/columns of square matrix sub(A). |
| d_d | Device | In | Pointer to an array of dimension `N`. |
| d_e | Device | In | Pointer to an array of dimension `N-1`. |
| d_Q | Device | In | Pointer into the local memory to an array of dimension `(LLD_Q, LOCc(JQ+N-1))`. |
| IQ | Host | In | Row index of the first row of the sub(Q). `IQ` must be a multiple of the row blocking dimension `MB_Q`. |
| JQ | Host | In | Column index of the first column of the sub(A). `JQ` must be a multiple of the column blocking dimension `NB_Q`. |
| descrQ | Host | In | Matrix descriptor associated to the global matrix Q. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | The size in bytes of the local device workspace needed by the routine [cusolverMpStedc()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc-label). |
| workspaceInBytesOnHost | Host | Out | The size in bytes of the local host workspace needed by [cusolverMpStedc()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

This routine currently supports the following combinations of data types:

| Data Type of Tridiagonal Matrix | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpOrmtr`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormtr "Link to this heading")

cusolverStatus_t cusolverMpOrmtr(
 cusolverMpHandle_t handle,
 cublasSideMode_t side,
 cublasFillMode_t uplo,
 cublasOperation_t trans,
 int64_t M,
 int64_t N,
 const void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 const void *d_tau,
 void *d_C,
 int64_t IC,
 int64_t JC,
 const cusolverMpMatrixDescriptor_t descrC,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Multiply distributed M-by-N matrix sub(C) denoting `C(IC:IC+M-1, JC:JC+N-1)` by the orthogonal matrix Q can be given from [cusolverMpSytrd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd-label).

The function can perform the following matrix product and overwrite the result on sub(C) for the side parameter `CUBLAS_SIDE_LEFT` and `CUBLAS_SIDE_RIGHT`.

The `op` can be translated to , ,  based on the `trans` argument `CUBLAS_OP_N`, `CUBLAS_OP_T` and `CUBLAS_OP_H`.

Q is an orthogonal matrix formed as the product of Householder reflectors as follows for the uplo parameter `CUBLAS_FILL_MODE_UPPER` and `CUBLAS_FILL_MODE_LOWER`

where `nq` is either `m` or `n` according to the side parameter of `CUBLAS_SIDE_LEFT` or `CUBLAS_SIDE_RIGHT` respectively.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| side | Host | In | Indicate that Q is applied from left or right side. |
| uplo | Host | In | Indicate that upper or lower triangular of sub(A) contains Householder reflectors. |
| trans | Host | In | Indicate that Q is applied with no-transpose or (conj)transpose. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. This array contains Householder reflectors below of diagonals with `tau` vector |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A).`JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_tau | Device | In | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. This array contains scalar factors of the Householder reflectors |
| d_C | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_C, LOCc(JC+N-1))`. On entry, the array contains the local pieces of the M-by-N distributed matrix sub(C). On exit, the sub(C) is overwritten by op(Q)*sub(C) or sub(C)*op(Q). |
| IC | Host | In | Row index of the first row of the sub(C). `IC` must be a multiple of the row blocking dimension `MB_C`. |
| JC | Host | In | Column index of the first column of the sub(C). `JC` must be a multiple of the column blocking dimension `NB_C`. |
| descrC | Host | In | Matrix descriptor associated to the global matrix C. |
|  |  |  |  |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpOrmtr_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormtr-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpOrmtr_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormtr-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

This function requires square block size `(MB_A == NB_A)` and alignment of sub(A) and sub(B) matrices, meaning `(MB_A == MB_C)` and `(IA == IC)`.

This routine supports the following combinations of data types:

| Data Type of A and C | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpOrmtr_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormtr-buffersize "Link to this heading")

cusolverStatus_t cusolverMpOrmtr_bufferSize(
 cusolverMpHandle_t handle,
 cublasSideMode_t side,
 cublasFillMode_t uplo,
 cublasOperation_t trans,
 int64_t M,
 int64_t N,
 const void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 const void *d_tau,
 void *d_C,
 int64_t IC,
 int64_t JC,
 const cusolverMpMatrixDescriptor_t descrC,
 cudaDataType_t computeType,
 size_t* workspaceInBytesOnDevice,
 size_t* workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpOrmtr()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormtr-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| side | Host | In | Indicate that Q is applied from left or right side. |
| uplo | Host | In | Indicate that upper or lower triangular of sub(A) contains Householder reflectors. |
| trans | Host | In | Indicate that Q is applied with no-transpose or (conj)transpose. |
| M | Host | In | Number of rows of sub(A). |
| N | Host | In | Number of columns of sub(A). |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. This array contains Householder reflectors below of diagonals with `tau` vector |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A).`JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_tau | Device | In | Pointer into the local memory to an array of dimension `LOCc(JA+N-1)`. This array contains scalar factors of the Householder reflectors |
| d_C | Device | In | Pointer into the local memory to an array of dimension `(LLD_C, LOCc(JC+N-1))`. On entry, the array contains the local pieces of the M-by-N distributed matrix sub(C). On exit, the sub(C) is overwritten by `op(Q)*sub(C)` or `sub(C)*op(Q)` based on the side input `CUBLAS_SIDE_LEFT` or `CUBLAS_SIDE_RIGHT` respectively. |
| IC | Host | In | Row index of the first row of the sub(C). `IC` must be a multiple of the row blocking dimension `MB_C`. |
| JC | Host | In | Column index of the first column of the sub(C).`JC` must be a multiple of the column blocking dimension `NB_C`. |
| descrC | Host | In | Matrix descriptor associated to the global matrix C. |
|  |  |  |  |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | The size in bytes of the local device workspace needed by [cusolverMpOrmtr()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormtr-label). |
| workspaceInBytesOnHost | Host | Out | The size in bytes of the local host workspace needed by [cusolverMpOrmtr()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormtr-label). |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpSyevd`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd "Link to this heading")

cusolverStatus_t cusolverMpSyevd(
 cusolverMpHandle_t handle,
 char *jobz,
 cublasFillMode_t uplo,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 void *d_d,
 void *d_Z,
 int64_t IZ,
 int64_t JZ,
 const cusolverMpMatrixDescriptor_t descrZ,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Computes all eigenvalues and optionally eigenvectors of a symmetric distributed N-by-N matrix sub(A) `A(IA:IA+N-1, JA:JA+N-1)` using the divide and conquer algorithm [cusolverMpStedc()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc-label). Note that the current implementation of the cusolverMpStedc may fail when the blocksize is not a power of two.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| jobz | Host | In | If `jobz = N`, then eigenvalues are computed and if `jobz = V`, then eigenvalues and eigenvectors are computed. |
| uplo | Host | In | Indicate that upper or lower triangular of sub(A) is used to compute eigen solutions. |
| N | Host | In | Number of rows and columns of sub(A). |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. This array contains local parts of the symmetric matrix A. |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A).`JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_d | Device | Out | Pointer into the memory to an array of global size `N`. On exit, this array contains real eigen values of the matrix A. |
| d_Z | Device | Out | Pointer into the local memory to an array of dimension `(LLD_Z, LOCc(JZ+N-1))`. On exit, the array local parts of orthonormal eigenvectors of the matrix A. |
| IZ | Host | In | Row index of the first row of the sub(Z). `IZ` must be a multiple of the row blocking dimension `MB_Z`. |
| JZ | Host | In | Column index of the first column of the sub(Z). `JZ` must be a multiple of the column blocking dimension `NB_Z`. |
| descrZ | Host | In | Matrix descriptor associated to the global matrix Z. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cusolverMpSyevd_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cusolverMpSyevd_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd-buffersize-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

This function requires square block size `(MB_A == NB_A)` and alignment of sub(A) and sub(B) matrices, meaning `(MB_A == MB_Z)` and `(IZ == IZ)`.

This routine supports the following combinations of data types:

| Data Type of A and C | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpSyevd_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd-buffersize "Link to this heading")

cusolverStatus_t cusolverMpSyevd_bufferSize(
 cusolverMpHandle_t handle,
 char *jobz,
 cublasFillMode_t uplo,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 const cusolverMpMatrixDescriptor_t descrA,
 void *d_d,
 void *d_Z,
 int64_t IZ,
 int64_t JZ,
 const cusolverMpMatrixDescriptor_t descrZ,
 cudaDataType_t computeType,
 size_t *workspaceInBytesOnDevice,
 size_t *workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpSyevd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| jobz | Host | In | If `jobz = N`, then eigenvalues are computed and if `jobz = V`, then eigenvalues and eigenvectors are computed. |
| uplo | Host | In | Indicate that upper or lower triangular of sub(A) is used to compute eigen solutions. |
| N | Host | In | Number of rows and columns of sub(A). |
| d_A | Device | In | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. This array contains local parts of the symmetric matrix A. |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A).`JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_d | Device | In | Pointer into the memory to an array of global size `N`. On exit, this array contains real eigen values of the matrix A. |
| d_Z | Device | In | Pointer into the local memory to an array of dimension `(LLD_Z, LOCc(JZ+N-1))`. On exit, the array local parts of orthonormal eigenvectors of the matrix A. |
| IZ | Host | In | Row index of the first row of the sub(Z). `IZ` must be a multiple of the row blocking dimension `MB_Z`. |
| JZ | Host | In | Column index of the first column of the sub(Z). `JZ` must be a multiple of the column blocking dimension `NB_Z`. |
| descrZ | Host | In | Matrix descriptor associated to the global matrix Z. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | The size in bytes of the local device workspace needed by [cusolverMpSyevd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd-label). |
| workspaceInBytesOnHost | Host | Out | The size in bytes of the local host workspace needed by [cusolverMpSyevd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd-label). |

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpSygst`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygst "Link to this heading")

cusolverStatus_t cusolverMpSygst(
 cusolverMpHandle_t handle,
 cusolverEigType_t ibtype,
 cublasFillMode_t uplo,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 const void *d_B,
 int64_t IB,
 int64_t JB,
 cusolverMpMatrixDescriptor_t descrB,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Reduces a hermitian-definite generalized eigenproblem to standard form. Denoting sub(A) and sub(B) as A(IA:IA+N-1, JA:JA+N-1) and B(IB:IB+N-1, JB:JB+N-1) respectively, the routine considers the following cases.

> *   `ibtype = CUSOLVER_EIG_TYPE_1`: the problem is sub(A)*x = lambda*sub(B)*x, and sub(A) is overwritten by inv(L)*sub(A)*inv(L^H) or inv(U^H)*sub(A)*inv(U).
> 
> *   `ibtype = CUSOLVER_EIG_TYPE_2 or 3`: the problem is sub(A)*sub(B)*x = lambda*x or sub(B)*sub(A)*x = lambda*x, and sub(A) is overwritten by L^H*sub(A)*L or U*sub(A)*U^H.

The sub(B) includes lower or upper Cholesky factors previously computed by [cusolverMpPotrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| ibtype | Host | In | Indicate the eigen problem type sub(A)*x=(lambda)*sub(B)*x, sub(A)*sub(B)x=(lambda)*x, or sub(B)*sub(A)*x=(lambda)*x. |
| uplo | Host | In | Indicate that lower `CUBLAS_FILL_MODE_LOWER` or upper `CUBLAS_FILL_MODE_UPPER` triangular of sub(A) and sub(B) are used to compute eigen solutions. |
| N | Host | In | Number of rows and columns of sub(A) and sub(B). |
| d_A | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. This array contains local parts of the symmetric matrix A. |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_B | Device | In | Pointer into the local memory to an array of dimension `(LLD_B, LOCc(JB+N-1))`. This array contains local parts of the symmetric matrix B. |
| IB | Host | In | Row index of the first row of the sub(B). `IB` must be a multiple of the row blocking dimension `MB_B`. |
| JB | Host | In | Column index of the first column of the sub(B). `JB` must be a multiple of the column blocking dimension `NB_B`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by [cusolverMpSygst()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygst-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by [cusolverMpSygst()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygst-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

The routine requires some alignment properties

> *   Same square blocksize is used `(MB == NB)` for the matrix A and B.
> 
> *   The beginning row and column of A and B are aligned each other i.e., `(IA == IB)` and `(JA == JB`.

Note that the current implementation supports the inputs of `ibtype = CUSOLVER_EIG_TYPE_1`, `uplo = CUBLAS_FILL_MODE_LOWER`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpSygst_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygst-buffersize "Link to this heading")

cusolverStatus_t cusolverMpSygst_bufferSize(
 cusolverMpHandle_t handle,
 cusolverEigType_t ibtype,
 cublasFillMode_t uplo,
 int64_t N,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 int64_t IB,
 int64_t JB,
 cusolverMpMatrixDescriptor_t descrB,
 cudaDataType_t computeType,
 size_t *workspaceInBytesOnDevice,
 size_t *workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpSygst()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygst-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| ibtype | Host | In | Indicate the eigen problem type sub(A)*x=(lambda)*sub(B)*x, sub(A)*sub(B)x=(lambda)*x, or sub(B)*sub(A)*x=(lambda)*x. |
| uplo | Host | In | Indicate that lower `CUBLAS_FILL_MODE_LOWER` or upper `CUBLAS_FILL_MODE_UPPER` triangular of sub(A) and sub(B) are used to compute eigen solutions. |
| N | Host | In | Number of rows and columns of sub(A) and sub(B). |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| IB | Host | In | Row index of the first row of the sub(B). `IB` must be a multiple of the row blocking dimension `MB_B`. |
| JB | Host | In | Column index of the first column of the sub(B). `JB` must be a multiple of the column blocking dimension `NB_B`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | The size in bytes of the local device workspace needed by [cusolverMpSygst()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygst-label). |
| workspaceInBytesOnHost | Host | Out | The size in bytes of the local host workspace needed by [cusolverMpSygst()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygst-label). |

The routine requires some alignment properties

> *   Same square blocksize is used `(MB == NB)` for the matrix A and B.
> 
> *   The beginning row and column of A and B are aligned each other i.e., `(IA == IB)` and `(JA == JB`.

Note that the current implementation supports the inputs of `ibtype = CUSOLVER_EIG_TYPE_1`, `uplo = CUBLAS_FILL_MODE_LOWER`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cusolverMpSygvd`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygvd "Link to this heading")

cusolverStatus_t cusolverMpSygvd(
 cusolverMpHandle_t handle,
 cusolverEigType_t ibtype,
 cusolverEigMode_t jobz,
 cublasFillMode_t uplo,
 int64_t N,
 void *d_A,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 void *d_B,
 int64_t IB,
 int64_t JB,
 cusolverMpMatrixDescriptor_t descrB,
 void *d_d,
 void *d_Z,
 int64_t IZ,
 int64_t JZ,
 cusolverMpMatrixDescriptor_t descrZ,
 cudaDataType_t computeType,
 void *d_work,
 size_t workspaceInBytesOnDevice,
 void *h_work,
 size_t workspaceInBytesOnHost,
 int *info)

Computes a hermitian-definite generalized eigenproblem using [cusolverMpSyevd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd-label). Denoting sub(A) and sub(B) as A(IA:IA+N-1, JA:JA+N-1) and B(IB:IB+N-1, JB:JB+N-1) respectively, the routine considers the following cases.

> *   `ibtype = CUSOLVER_EIG_TYPE_1`: the problem is sub(A)*x = lambda*sub(B)*x.
> 
> *   `ibtype = CUSOLVER_EIG_TYPE_2`: the problem is sub(A)*sub(B)*x = lambda*x.
> 
> *   `ibtype = CUSOLVER_EIG_TYPE_3`: the problem is sub(B)*sub(A)*x = lambda*x.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| ibtype | Host | In | Indicate the eigen problem type sub(A)*x=(lambda)*sub(B)*x, sub(A)*sub(B)x=(lambda)*x, or sub(B)*sub(A)*x=(lambda)*x. |
| jobz | Host | In | Indicate whether the routine computes eigenvalues only `CUSOLVER_EIG_MODE_NOVECTOR` or includes eigenvectors as well `CUSOLVER_EIG_MODE_VECTOR`. |
| uplo | Host | In | Indicate that lower `CUBLAS_FILL_MODE_LOWER` or upper `CUBLAS_FILL_MODE_UPPER` triangular of sub(A) and sub(B) are used to compute eigen solutions. |
| N | Host | In | Number of rows and columns of sub(A) and sub(B). |
| d_A | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_A, LOCc(JA+N-1))`. This array contains local parts of the symmetric matrix A and will be overwritten with standard eigen problem. |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| d_B | Device | In/Out | Pointer into the local memory to an array of dimension `(LLD_B, LOCc(JB+N-1))`. This array contains local parts of the symmetric matrix B and will be overwritten with Cholesky factors. |
| IB | Host | In | Row index of the first row of the sub(B). `IB` must be a multiple of the row blocking dimension `MB_B`. |
| JB | Host | In | Column index of the first column of the sub(B). `JB` must be a multiple of the column blocking dimension `NB_B`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
| d_d | Device | Out | Pointer into the memory to an array of global size `N`. On exit, this array contains real eigen values of the matrix A. |
| d_Z | Device | Out | Pointer into the local memory to an array of dimension `(LLD_Z, LOCc(JZ+N-1))`. On exit, the array local parts of orthonormal eigenvectors of the matrix A. |
| IZ | Host | In | Row index of the first row of the sub(Z). `IZ` must be a multiple of the row blocking dimension `MB_Z`. |
| JZ | Host | In | Column index of the first column of the sub(Z). `JZ` must be a multiple of the column blocking dimension `NB_Z`. |
| descrZ | Host | In | Matrix descriptor associated to the global matrix Z. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | Out | The size in bytes of the local device workspace needed by [cusolverMpSygvd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygvd-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | Out | The size in bytes of the local host workspace needed by [cusolverMpSygvd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygvd-label). |
| info | Device | Out | `info < 0` indicates an incorrect value of the i-th argument of the function. |

The routine requires some alignment properties

> *   Same square blocksize is used `(MB == NB)` for the matrix A, B, and Z.
> 
> *   The beginning row and column of A, B and Z are aligned each other i.e., `(IA == IB == IZ)` and `(JA == JB == JZ`.

Note that the current implementation supports the inputs of `ibtype = CUSOLVER_EIG_TYPE_1`, `jobz = CUSOLVER_EIG_MODE_VECTOR`, `uplo = CUBLAS_FILL_MODE_LOWER`.

This routine supports the following combinations of data types:

| Data Type of A and C | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cusolverMpSygvd_bufferSize`[#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygvd-buffersize "Link to this heading")

cusolverStatus_t cusolverMpSygvd_bufferSize(
 cusolverMpHandle_t handle,
 cusolverEigType_t ibtype,
 cusolverEigMode_t jobz,
 cublasFillMode_t uplo,
 int64_t N,
 int64_t IA,
 int64_t JA,
 cusolverMpMatrixDescriptor_t descrA,
 int64_t IB,
 int64_t JB,
 cusolverMpMatrixDescriptor_t descrB,
 int64_t IZ,
 int64_t JZ,
 cusolverMpMatrixDescriptor_t descrZ,
 cudaDataType_t computeType,
 size_t *workspaceInBytesOnDevice,
 size_t *workspaceInBytesOnHost)

Computes the size in bytes of the host and device working buffers required by [cusolverMpSygvd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygvd-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuSOLVERMp library handle. |
| ibtype | Host | In | Indicate the eigen problem type sub(A)*x=(lambda)*sub(B)*x, sub(A)*sub(B)x=(lambda)*x, or sub(B)*sub(A)*x=(lambda)*x. |
| jobz | Host | In | Indicate whether the routine computes eigenvalues only `CUSOLVER_EIG_MODE_NOVECTOR` or includes eigenvectors as well `CUSOLVER_EIG_MODE_VECTOR`. |
| uplo | Host | In | Indicate that lower `CUBLAS_FILL_MODE_LOWER` or upper `CUBLAS_FILL_MODE_UPPER` triangular of sub(A) and sub(B) are used to compute eigen solutions. |
| N | Host | In | Number of rows and columns of sub(A) and sub(B). |
| IA | Host | In | Row index of the first row of the sub(A). `IA` must be a multiple of the row blocking dimension `MB_A`. |
| JA | Host | In | Column index of the first column of the sub(A). `JA` must be a multiple of the column blocking dimension `NB_A`. |
| descrA | Host | In | Matrix descriptor associated to the global matrix A. |
| IB | Host | In | Row index of the first row of the sub(B). `IB` must be a multiple of the row blocking dimension `MB_B`. |
| JB | Host | In | Column index of the first column of the sub(B). `JB` must be a multiple of the column blocking dimension `NB_B`. |
| descrB | Host | In | Matrix descriptor associated to the global matrix B. |
| IZ | Host | In | Row index of the first row of the sub(Z). `IZ` must be a multiple of the row blocking dimension `MB_Z`. |
| JZ | Host | In | Column index of the first column of the sub(Z). `JZ` must be a multiple of the column blocking dimension `NB_Z`. |
| descrZ | Host | In | Matrix descriptor associated to the global matrix Z. |
| computeType | Host | In | Data type used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | The size in bytes of the local device workspace needed by [cusolverMpSygvd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygvd-label). |
| workspaceInBytesOnHost | Host | Out | The size in bytes of the local host workspace needed by [cusolverMpSygvd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygvd-label). |

The routine requires some alignment properties

> *   Same square blocksize is used `(MB == NB)` for the matrix A, B, and Z.
> 
> *   The beginning row and column of A, B and Z are aligned each other i.e., `(IA == IB == IZ)` and `(JA == JB == JZ`.

Note that the current implementation supports the inputs of `ibtype = CUSOLVER_EIG_TYPE_1`, `jobz = CUSOLVER_EIG_MODE_VECTOR`, `uplo = CUBLAS_FILL_MODE_LOWER`.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

Links/Buttons:
- [](https://docs.nvidia.com/cuda/cusolvermp/index.html)
- [How to Use cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/usage/index.html.md)
- [#](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygvd-buffersize)
- [cusolverMpHandle_t](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermphandle-t-label)
- [cusolverMpCreate()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpcreate-label)
- [cusolverStatus_t](https://docs.nvidia.com/cuda/cusolver/index.html.md#cusolverstatus-t)
- [cusolverMathMode_t](https://docs.nvidia.com/cuda/cusolver/index.html.md#cusolvermathmode-t)
- [cublasEmulationStrategy_t](https://docs.nvidia.com/cuda/cublas/index.html.md#cublasemulationstrategy-t)
- [grid](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermpgrid-t-label)
- [cusolverMpMatrixDescriptor_t](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermpmatrixdescriptor-t-label)
- [cusolverMpLoggerCallback_t](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md#cusolvermploggercallback-t-label)
- [cusolverMpGetrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-label)
- [cusolverMpGetrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs-label)
- [cusolverMpGetrf_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrf-buffersize-label)
- [cusolverMpGetrs_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgetrs-buffersize-label)
- [cusolverMpPotrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-label)
- [cusolverMpPotrs()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs-label)
- [cusolverMpPotrf_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrf-buffersize-label)
- [cusolverMpPotrs_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermppotrs-buffersize-label)
- [cusolverMpGeqrf_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-buffersize-label)
- [cusolverMpGeqrf()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgeqrf-label)
- [cusolverMpOrmqr_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormqr-buffersize-label)
- [cusolverMpOrmqr()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormqr-label)
- [cusolverMpGels_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgels-buffersize-label)
- [cusolverMpGels()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpgels-label)
- [cusolverMpSytrd_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd-buffersize-label)
- [cusolverMpSytrd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsytrd-label)
- [cusolverMpStedc_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc-buffersize-label)
- [cusolverMpStedc()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpstedc-label)
- [cusolverMpOrmtr_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormtr-buffersize-label)
- [cusolverMpOrmtr()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpormtr-label)
- [cusolverMpSyevd_bufferSize()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd-buffersize-label)
- [cusolverMpSyevd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsyevd-label)
- [cusolverMpSygst()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygst-label)
- [cusolverMpSygvd()](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.md#cusolvermpsygvd-label)
- [previous cuSOLVERMp Data Types](https://docs.nvidia.com/cuda/cusolvermp/usage/types.html.md)
- [next Release Notes](https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html.md)
