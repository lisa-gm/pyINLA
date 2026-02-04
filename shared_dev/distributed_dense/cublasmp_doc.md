Title: cuBLASMp C API — cuBLASMp

URL Source: https://docs.nvidia.com/cuda/cublasmp/usage/functions.html

Published Time: Thu, 11 Dec 2025 19:23:36 GMT

Markdown Content:
cuBLASMp C API[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmp-c-api "Link to this heading")
-------------------------------------------------------------------------------------------------------------------

Library Management[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#library-management "Link to this heading")
---------------------------------------------------------------------------------------------------------------------------

### `cublasMpCreate`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpcreate "Link to this heading")

cublasMpStatus_t cublasMpCreate(
 cublasMpHandle_t *handle,
 cudaStream_t stream);

This function initializes the cuBLASMp library handle ([cublasMpHandle_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmphandle-t-label)) which holds the cuBLASMp library context. It allocates light hardware resources on the host, and must be called prior to making any other cuBLASMp library calls.

Calling any cuBLASMp function which uses [cublasMpHandle_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmphandle-t-label) without a previous call of [cublasMpCreate()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpcreate-label) will return an error.

The cuBLASMp library context is tied to the current CUDA device and the given CUDA stream.

Sharing a device with multiple processes may result in undefined behavior.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | Out | cuBLASMp library handle. |
| stream | Host | In | Stream that will be assigned to the handle. |

### `cublasMpDestroy`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpdestroy "Link to this heading")

cublasMpStatus_t cublasMpDestroy(
 cublasMpHandle_t handle);

This function destroys the cuBLASMp library handle ([cublasMpHandle_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmphandle-t-label)) which holds the cuBLASMp library context.

The cuBLASMp library context is tied to the CUDA device that was set when calling [cublasMpCreate()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpcreate-label). Only one handle per process and per GPU supported.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In/Out | cuBLASMp library handle to destroy. |

### `cublasMpStreamSet`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpstreamset "Link to this heading")

cublasMpStatus_t cublasMpStreamSet(
 cublasMpHandle_t handle,
 cudaStream_t stream);

This function sets the CUDA stream to be used in the computations.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| stream | Host | In | CUDA stream pointer to set. |

### `cublasMpStreamGet`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpstreamget "Link to this heading")

cublasMpStatus_t cublasMpStreamGet(
 cublasMpHandle_t handle,
 cudaStream_t* stream);

This function returns the current CUDA stream that is being used in the computations.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| stream | Host | Out | CUDA stream pointer to set. |

### `cublasMpSetEmulationStrategy`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsetemulationstrategy "Link to this heading")

cublasMpStatus_t cublasMpSetEmulationStrategy(
 cublasMpHandle_t handle,
 cublasMpEmulationStrategy_t emulationStrategy);

This function allows you to select how the library should make use of floating point emulation. For more details, please see [cublasMpEmulationStrategy_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpemulationstrategy-t-label).

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| emulationStrategy | Host | In | Emulation strategy to use. See [cublasMpEmulationStrategy_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpemulationstrategy-t-label). |

### `cublasMpGetEmulationStrategy`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgetemulationstrategy "Link to this heading")

cublasMpStatus_t cublasMpGetEmulationStrategy(
 cublasMpHandle_t handle,
 cublasMpEmulationStrategy_t* emulationStrategy);

This function retrieves the current emulation strategy from the cuBLASMp library handle.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| emulationStrategy | Host | Out | Pointer to receive the current emulation strategy. See [cublasMpEmulationStrategy_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpemulationstrategy-t-label). |

### `cublasMpGetVersion`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgetversion "Link to this heading")

cublasMpStatus_t cublasMpGetVersion(
 int *version);

This function returns the version number of the cuBLASMp library.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| version | Host | Out | cuBLASMp library version. Value is `CUBLASMP_VER_MAJOR * 1000 + CUBLASMP_VER_MINOR * 100 + CUBLASMP_VER_PATCH`. |

* * *

Grid Management[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#grid-management "Link to this heading")
---------------------------------------------------------------------------------------------------------------------

### `cublasMpGridCreate`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgridcreate "Link to this heading")

cublasMpStatus_t cublasMpGridCreate(
 int64_t nprow,
 int64_t npcol,
 cublasMpGridLayout_t layout,
 ncclComm_t comm,
 cublasMpGrid_t* grid);

This function initializes the [grid](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpgrid-t-label) opaque data structure. It maps the given resources (communicator, grid dimensions and [grid layout](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpgridlayout-t-label)) to a grid object.

All the processes defined to be in this grid must enter this function.

Note

cuBLASMp will initialize NVSHMEM as the first grid is created, hence the user should ensure that it uses a communicator that contains all the required ranks. If NVSHMEM was previously initialized by the user in their application, the first cuBLASMp grid should be created using the same set of ranks. cuBLASMp will call `nvshmem_finalize` as part of the [cublasMpGridDestroy()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgriddestroy-label) call of the last remaining grid.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| nprow | Host | In | How many row processes the grid contains. |
| npcol | Host | In | How many column processes the grid contains. |
| layout | Host | In | Grid’s layout ([cublasMpGridLayout_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpgridlayout-t-label)). |
| comm | Host | In | Communicator associated with the grid. |
| grid | Host | In/Out | Pointer to a grid object. |

* * *

### `cublasMpGridDestroy`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgriddestroy "Link to this heading")

cublasMpStatus_t cublasMpGridDestroy(
 cublasMpGrid_t grid);

This function destroys the given `grid` object.

All the processes defined to be in this grid must enter this function.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| grid | Host | In/Out | Grid object to destroy. |

* * *

Matrix Management[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#matrix-management "Link to this heading")
-------------------------------------------------------------------------------------------------------------------------

### `cublasMpMatrixDescriptorCreate`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatrixdescriptorcreate "Link to this heading")

cublasMpStatus_t cublasMpMatrixDescriptorCreate(
 int64_t m,
 int64_t n,
 int64_t mb,
 int64_t nb,
 int64_t rsrc,
 int64_t csrc,
 int64_t lld,
 cudaDataType_t type,
 cublasMpGrid_t grid,
 cublasMpMatrixDescriptor_t* desc);

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| m | Host | In | Number of rows in the global matrix. |
| n | Host | In | Number of columns in the global matrix. |
| mb | Host | In | Blocking factor used to distribute the rows of the global matrix. |
| nb | Host | In | Blocking factor used to distribute the columns of the global matrix. |
| rsrc | Host | In | Row rank of the process who owns the first row block of the global matrix. |
| csrc | Host | In | Column rank of the process who owns the first column block of the global matrix. |
| lld | Host | In | Leading dimension of the local matrix. |
| type | Host | In | Data type of the matrix. |
| grid | Host | In | Grid object associated with the matrix descriptor. |
| desc | Host | Out | Matrix descriptor object initialized by this function. |

Supported values for `type` argument are listed below:

| Data Type | Description |
| --- | --- |
| CUDA_R_8I | 8-bit real signed integer. |
| CUDA_R_32I | 32-bit real signed integer. |
| CUDA_R_4F_E2M1 | 4-bit real floating point in E2M1 format. |
| CUDA_R_8F_E4M3 | 8-bit real floating point in E4M3 format. |
| CUDA_R_8F_E5M2 | 8-bit real floating point in E5M2 format. |
| CUDA_R_16F | 16-bit real half precision floating-point. |
| CUDA_R_16BF | 16-bit real bfloat16 floating-point. |
| CUDA_R_32F | 32-bit real single precision floating-point. |
| CUDA_R_64F | 64-bit real double precision floating-point. |
| CUDA_C_32F | 64-bit structure comprised of two single precision floating-points representing a complex number. |
| CUDA_C_64F | 128-bit structure comprised of two double precision floating-points representing a complex number. |

* * *

### `cublasMpMatrixDescriptorDestroy`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatrixdescriptordestroy "Link to this heading")

cublasMpStatus_t cublasMpMatrixDescriptorDestroy(
 cublasMpMatrixDescriptor_t desc);

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| desc | Host | In/Out | Matrix descriptor object to destroy. |

### `cublasMpMatrixDescriptorInit`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatrixdescriptorinit "Link to this heading")

cublasMpStatus_t cublasMpMatrixDescriptorInit(
 int64_t m,
 int64_t n,
 int64_t mb,
 int64_t nb,
 int64_t rsrc,
 int64_t csrc,
 int64_t lld,
 cudaDataType_t type,
 cublasMpGrid_t grid,
 cublasMpMatrixDescriptor_t desc);

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| m | Host | In | Number of rows in the global matrix. |
| n | Host | In | Number of columns in the global matrix. |
| mb | Host | In | Blocking factor used to distribute the rows of the global matrix. |
| nb | Host | In | Blocking factor used to distribute the columns of the global matrix. |
| rsrc | Host | In | Row rank of the process who owns the first row block of the global matrix. |
| csrc | Host | In | Column rank of the process who owns the first column block of the global matrix. |
| lld | Host | In | Leading dimension of the local matrix. |
| type | Host | In | Data type of the matrix. |
| grid | Host | In | Grid object associated with the matrix descriptor. |
| desc | Host | In/Out | Matrix descriptor object initialized by this function. |

Supported values for `type` argument are listed below:

| Data Type | Description |
| --- | --- |
| CUDA_R_8I | 8-bit real signed integer. |
| CUDA_R_32I | 32-bit real signed integer. |
| CUDA_R_4F_E2M1 | 4-bit real floating point in E2M1 format. |
| CUDA_R_8F_E4M3 | 8-bit real floating point in E4M3 format. |
| CUDA_R_8F_E5M2 | 8-bit real floating point in E5M2 format. |
| CUDA_R_16F | 16-bit real half precision floating-point. |
| CUDA_R_16BF | 16-bit real bfloat16 floating-point. |
| CUDA_R_32F | 32-bit real single precision floating-point. |
| CUDA_R_64F | 64-bit real double precision floating-point. |
| CUDA_C_32F | 64-bit structure comprised of two single precision floating-points representing a complex number. |
| CUDA_C_64F | 128-bit structure comprised of two double precision floating-points representing a complex number. |

* * *

Matmul Properties[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#matmul-properties "Link to this heading")
-------------------------------------------------------------------------------------------------------------------------

### `cublasMpMatmulDescriptorCreate`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmuldescriptorcreate "Link to this heading")

cublasMpStatus_t cublasMpMatmulDescriptorCreate(
 cublasMpMatmulDescriptor_t* matmulDesc,
 cublasComputeType_t computeType);

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| matmulDesc | Host | In/Out | Pointer to a cublasMpMatmulDescriptor object to initialize. |
| computeType | Host | In | [cuBLAS compute type](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t) used for computations. See table below for supported combinations. |

Supported values for `computeType` argument are listed below:

| Compute Types |
| --- |
| CUBLAS_COMPUTE_32I |
| CUBLAS_COMPUTE_32I_PEDANTIC |
| CUBLAS_COMPUTE_16F |
| CUBLAS_COMPUTE_16F_PEDANTIC |
| CUBLAS_COMPUTE_32F |
| CUBLAS_COMPUTE_32F_PEDANTIC |
| CUBLAS_COMPUTE_32F_FAST_16F |
| CUBLAS_COMPUTE_32F_FAST_16BF |
| CUBLAS_COMPUTE_32F_FAST_TF32 |
| CUBLAS_COMPUTE_64F |
| CUBLAS_COMPUTE_64F_PEDANTIC |

### `cublasMpMatmulDescriptorDestroy`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmuldescriptordestroy "Link to this heading")

cublasMpStatus_t cublasMpMatmulDescriptorDestroy(
 cublasMpMatmulDescriptor_t matmulDesc);

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| matmulDesc | Host | In/Out | Matmul descriptor object to destroy. |

### `cublasMpMatmulDescriptorAttributeSet`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmuldescriptorattributeset "Link to this heading")

cublasMpStatus_t cublasMpMatmulDescriptorAttributeSet(
 cublasMpMatmulDescriptor_t matmulDesc,
 cublasMpMatmulDescriptorAttribute_t attr,
 const void* buf,
 size_t sizeInBytes);

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| matmulDesc | Host | In | Matmul descriptor object to set its attribute. |
| attr | Host | In | Matmul descriptor attribute to set. |
| buf | Host | In | Attribute value to set. |
| sizeInBytes | Host | In | Attribute buffer size in bytes. |

### `cublasMpMatmulDescriptorAttributeGet`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmuldescriptorattributeget "Link to this heading")

cublasMpStatus_t cublasMpMatmulDescriptorAttributeGet(
 cublasMpMatmulDescriptor_t matmulDesc,
 cublasMpMatmulDescriptorAttribute_t attr,
 const void* buf,
 size_t sizeInBytes,
 size_t* sizeWritten);

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| matmulDesc | Host | In | Matmul descriptor object to set its attribute. |
| attr | Host | In | Matmul descriptor attribute to set. |
| buf | Host | Out | Attribute value to set. |
| sizeInBytes | Host | In | Attribute buffer size in bytes. |
| sizeWritten | Host | Out | Size of the attribute written into `buf` in bytes. |

* * *

Utility[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#utility "Link to this heading")
-----------------------------------------------------------------------------------------------------

### `cublasMpNumroc`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpnumroc "Link to this heading")

int64_t cublasMpNumroc(
 int64_t n,
 int64_t nb,
 uint32_t iproc,
 uint32_t isrcproc,
 uint32_t nprocs);

Computes the number of rows or columns of a distributed matrix owned by the process indicated by `iproc` argument.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| n | Host | In | Number of rows or columns in the global distributed matrix. |
| nb | Host | In | Row or column blocking size of the global matrix. |
| iproc | Host | In | The coordinate of the process whose local array row or column is to be determined. |
| isrcproc | Host | In | The coordinate of the process that owns the first row or column of the distributed matrix. |
| nprocs | Host | In | The total number of row or column processes over which the matrix is distributed. |

* * *

### `cublasMpGemr2D`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemr2d "Link to this heading")

cublasMpStatus_t cublasMpGemr2D(
 cublasMpHandle_t handle,
 int64_t m,
 int64_t n,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 void* d_work,
 size_t workspaceSizeInBytesOnDevice,
 void* h_work,
 size_t workspaceSizeInBytesOnHost,
 ncclComm_t global_comm);

This function redistributes general rectangular matrix A according to the distribution properties of matrix B.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| m | Host | In | Number of rows of sub(A) and sub(B). |
| n | Host | In | Number of columns of sub(A) and sub(B). |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). |
| ja | Host | In | Column index of the first column of the sub(A). |
| descA | Host | In | Matrix descriptor associated to the global matrix A. descA’s grid value must be set to null in processes that are not part of the grid of A. |
| b | Device | Out | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). |
| jb | Host | In | Column index of the first column of the sub(B). |
| descB | Host | In | Matrix descriptor associated to the global matrix B. descB’s grid value must be set to null in processes that are not part of the grid of B. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cublasMpGemr2D_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemr2d-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cublasMpGemr2D_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemr2d-buffersize-label). |
| global_comm | Host | In | A communicator containing at least the union of all processes in the communicators of A and B. All processes in the communicator must call this function, even if they do not own a piece of either matrix. |

* * *

### `cublasMpGemr2D_bufferSize`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemr2d-buffersize "Link to this heading")

cublasMpStatus_t cublasMpGemr2D_bufferSize(
 cublasMpHandle_t handle,
 int64_t m,
 int64_t n,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 size_t* workspaceSizeInBytesOnDevice,
 size_t* workspaceSizeInBytesOnHost,
 ncclComm_t global_comm);

This function returns the required buffer sizes to perform [cublasMpGemr2D()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemr2d-label) on the given input.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| m | Host | In | Number of rows of sub(A) and sub(B). |
| n | Host | In | Number of columns of sub(A) and sub(B). |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). |
| ja | Host | In | Column index of the first column of the sub(A). |
| descA | Host | In | Matrix descriptor associated to the global matrix A. descA’s grid value must be set to null in processes that are not part of the grid of A. |
| b | Device | In | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). |
| jb | Host | In | Column index of the first column of the sub(B). |
| descB | Host | In | Matrix descriptor associated to the global matrix B. descB’s grid value must be set to null in processes that are not part of the grid of B. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cublasMpGemr2D()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemr2d-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cublasMpGemr2D()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemr2d-label). |
| global_comm | Host | In | A communicator containing at least the union of all processes in the communicators of A and B. All processes in the communicator must call this function, even if they do not own a piece of either matrix. |

* * *

### `cublasMpTrmr2D`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrmr2d "Link to this heading")

cublasMpStatus_t cublasMpTrmr2D(
 cublasMpHandle_t handle,
 cublasFillMode_t uplo,
 cublasDiagType_t diag,
 int64_t m,
 int64_t n,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 void* d_work,
 size_t workspaceSizeInBytesOnDevice,
 void* h_work,
 size_t workspaceSizeInBytesOnHost,
 ncclComm_t global_comm);

This function redistributes trapezoidal matrix A according to the distribution properties of trapezoidal matrix B.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| uplo | Host | In | Indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements. |
| diag | Host | In | Indicates if the elements on the main diagonal of matrix A are unity and should not be accessed. |
| m | Host | In | Number of rows of sub(A) and sub(B). |
| n | Host | In | Number of columns of sub(A) and sub(B). |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). |
| ja | Host | In | Column index of the first column of the sub(A). |
| descA | Host | In | Matrix descriptor associated to the global matrix A. descA’s grid value must be set to null in processes that are not part of the grid of A. |
| b | Device | Out | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). |
| jb | Host | In | Column index of the first column of the sub(B). |
| descB | Host | In | Matrix descriptor associated to the global matrix B. descB’s grid value must be set to null in processes that are not part of the grid of B. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cublasMpTrmr2D_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrmr2d-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cublasMpTrmr2D_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrmr2d-buffersize-label). |
| global_comm | Host | In | A communicator containing at least the union of all processes in the communicators of A and B. All processes in the communicator must call this function, even if they do not own a piece of either matrix. |

* * *

### `cublasMpTrmr2D_bufferSize`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrmr2d-buffersize "Link to this heading")

cublasMpStatus_t cublasMpTrmr2D_bufferSize(
 cublasMpHandle_t handle,
 cublasFillMode_t uplo,
 cublasDiagType_t diag,
 int64_t m,
 int64_t n,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 size_t* workspaceSizeInBytesOnDevice,
 size_t* workspaceSizeInBytesOnHost,
 ncclComm_t global_comm);

This function returns the required buffer sizes to perform [cublasMpTrmr2D()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrmr2d-label) on the given input.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| uplo | Host | In | Indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements. |
| diag | Host | In | Indicates if the elements on the main diagonal of matrix A are unity and should not be accessed. |
| m | Host | In | Number of rows of sub(A) and sub(B). |
| n | Host | In | Number of columns of sub(A) and sub(B). |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). |
| ja | Host | In | Column index of the first column of the sub(A). |
| descA | Host | In | Matrix descriptor associated to the global matrix A. descA’s grid value must be set to null in processes that are not part of the grid of A. |
| b | Device | In | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). |
| jb | Host | In | Column index of the first column of the sub(B). |
| descB | Host | In | Matrix descriptor associated to the global matrix B. descB’s grid value must be set to null in processes that are not part of the grid of B. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cublasMpTrmr2D()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrmr2d-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cublasMpTrmr2D()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrmr2d-label). |
| global_comm | Host | In | A communicator containing at least the union of all processes in the communicators of A and B. All processes in the communicator must call this function, even if they do not own a piece of either matrix. |

* * *

Logging[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#logging "Link to this heading")
-----------------------------------------------------------------------------------------------------

### `cublasMpLoggerSetCallback`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmploggersetcallback "Link to this heading")

cublasMpStatus_t cublasMpLoggerSetCallback(
 cublasMpLoggerCallback_t callback);

This function sets the logging callback function.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| callback | Host | In | Pointer to a callback function. See [cublasMpLoggerCallback_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmploggercallback-t-label). |

Warning

This is an experimental feature.

* * *

### `cublasMpLoggerSetFile`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmploggersetfile "Link to this heading")

cublasMpStatus_t cublasMpLoggerSetFile(
 FILE *file);

This function sets the logging output file. Note: once registered using this function call, the provided file handle must not be closed unless the function is called again to switch to a different file handle.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| file | Host | In | Pointer to an open file. File should have write permission. |

Warning

This is an experimental feature.

* * *

### `cublasMpLoggerOpenFile`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmploggeropenfile "Link to this heading")

cublasMpStatus_t cublasMpLoggerOpenFile(
 const char* logFile);

This function opens a logging output file in the given path.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| logFile | Host | In | Path of the logging output file. |

Warning

This is an experimental feature.

* * *

### `cublasMpLoggerSetLevel`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmploggersetlevel "Link to this heading")

cublasMpStatus_t cublasMpLoggerSetLevel(
 int level);

This function sets the logging level.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| level | Host | In | Value of the logging level. See [cuBLASMp Logging](https://docs.nvidia.com/cuda/cublasmp/usage/logging.html.md#logging-label). |

Warning

This is an experimental feature.

* * *

### `cublasMpLoggerSetMask`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmploggersetmask "Link to this heading")

cublasMpStatus_t cublasMpLoggerSetMask(
 int mask);

This function sets the value of the logging mask.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| mask | Host | In | Value of the logging mask. See [cuBLASMp Logging](https://docs.nvidia.com/cuda/cublasmp/usage/logging.html.md#logging-label). |

Warning

This is an experimental feature.

* * *

### `cublasMpLoggerForceDisable`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmploggerforcedisable "Link to this heading")

cublasMpStatus_t cublasMpLoggerForceDisable();

This function disables logging for the entire run.

Warning

This is an experimental feature.

* * *

Dense Linear Algebra APIs[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#dense-linear-algebra-apis "Link to this heading")
-----------------------------------------------------------------------------------------------------------------------------------------

### `cublasMpTrsm`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrsm "Link to this heading")

cublasMpStatus_t cublasMpTrsm(
 cublasMpHandle_t handle,
 cublasSideMode_t side,
 cublasFillMode_t uplo,
 cublasOperation_t trans,
 cublasDiagType_t diag,
 int64_t m,
 int64_t n,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 cublasComputeType_t computeType,
 void* d_work,
 size_t workspaceSizeInBytesOnDevice,
 void* h_work,
 size_t workspaceSizeInBytesOnHost);

This function solves the triangular linear system with multiple right-hand-sides

where  is a triangular matrix stored in lower or upper mode with or without the main diagonal,  and  are  matrices, and  is a scalar. Also, for matrix

The solution  overwrites the right-hand-sides  on exit.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| side | Host | In | Indicates if matrix A is on the left or right of X. |
| uplo | Host | In | Indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements. |
| trans | Host | In | Operation op(A) that is non- or (conj.) transpose. |
| diag | Host | In | Indicates if the elements on the main diagonal of matrix A are unity and should not be accessed. |
| m | Host | In | Number of rows of matrix sub(B), with matrix sub(A) sized accordingly. |
| n | Host | In | Number of columns of matrix sub(B), with matrix sub(A) is sized accordingly. |
| alpha | Host | In | Scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| b | Device | In/Out | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). `ib` must be a multiple of the row blocking dimension `mbB`. |
| jb | Host | In | Column index of the first column of the sub(B). `jb` must be a multiple of the column blocking dimension `nbB`. |
| descB | Host | In | Matrix descriptor associated to the global matrix B. |
| computeType | Host | In | [cuBLAS compute type](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t) used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cublasMpTrsm_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrsm-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cublasMpTrsm_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrsm-buffersize-label). |

This function requires square block size.

This routine supports the following combinations of data types:

| Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype |
| --- | --- | --- | --- |
| CUBLAS_COMPUTE_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_64F | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

The `computeType` parameter provided to this function is used only for internal matrix-matrix multiplications.

* * *

#### `cublasMpTrsm_bufferSize`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrsm-buffersize "Link to this heading")

cublasMpStatus_t cublasMpTrsm_bufferSize(
 cublasMpHandle_t handle,
 cublasSideMode_t side,
 cublasFillMode_t uplo,
 cublasOperation_t trans,
 cublasDiagType_t diag,
 int64_t m,
 int64_t n,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 cublasComputeType_t computeType,
 size_t* workspaceSizeInBytesOnDevice,
 size_t* workspaceSizeInBytesOnHost);

This function returns the required buffer sizes to perform [cublasMpTrsm()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrsm-label) on the given input.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| side | Host | In | Indicates if matrix A is on the left or right of X. |
| uplo | Host | In | Indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements. |
| trans | Host | In | Operation op(A) that is non- or (conj.) transpose. |
| diag | Host | In | Indicates if the elements on the main diagonal of matrix A are unity and should not be accessed. |
| m | Host | In | Number of rows of matrix sub(B), with matrix sub(A) sized accordingly. |
| n | Host | In | Number of columns of matrix sub(B), with matrix sub(A) is sized accordingly. |
| alpha | Host | In | Scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| b | Device | In | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). `ib` must be a multiple of the row blocking dimension `mbB`. |
| jb | Host | In | Column index of the first column of the sub(B). `jb` must be a multiple of the column blocking dimension `nbB`. |
| descB | Host | In | Matrix descriptor associated to the global matrix B. |
| computeType | Host | In | [cuBLAS compute type](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t) used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cublasMpTrsm()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrsm-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cublasMpTrsm()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrsm-label). |

This function requires square block size.

This routine supports the following combinations of data types:

| Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype |
| --- | --- | --- | --- |
| CUBLAS_COMPUTE_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_64F | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

The `computeType` parameter provided to this function is used only for internal matrix-matrix multiplications.

* * *

### `cublasMpGemm`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemm "Link to this heading")

cublasMpStatus_t cublasMpGemm(
 cublasMpHandle_t handle,
 cublasOperation_t transA,
 cublasOperation_t transB,
 int64_t m,
 int64_t n,
 int64_t k,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 const void* beta,
 void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 cublasComputeType_t computeType,
 void* d_work,
 size_t workspaceSizeInBytesOnDevice,
 void* h_work,
 size_t workspaceSizeInBytesOnHost);

This function performs the matrix-matrix multiplication

where  and  are scalars, and  ,  and  are matrices stored in column-major format with dimensions  ,  and  , respectively. Also, for matrix

and  is defined similarly for matrix  .

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| transA | Host | In | Operation op(A) that is non- or (conj.) transpose. |
| transB | Host | In | Operation op(B) that is non- or (conj.) transpose. |
| m | Host | In | Number of rows of sub(A) and sub(C). |
| n | Host | In | Number of columns of sub(B) and sub(C). |
| k | Host | In | Number of columns of sub(A) and rows of sub(B). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| b | Device | In | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). `ib` must be a multiple of the row blocking dimension `mbB`. |
| jb | Host | In | Column index of the first column of the sub(B). `jb` must be a multiple of the column blocking dimension `nbB`. |
| descB | Host | In | Matrix descriptor associated to the global matrix B. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In/Out | Pointer to the first entry of the local portion of the global matrix C. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| computeType | Host | In | [cuBLAS compute type](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t) used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cublasMpGemm_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemm-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cublasMpGemm_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemm-buffersize-label). |

Note

This routine will internally call [cublasMpMatmul()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul-label) with `d == c`.

This routine supports the following combinations of data types:

| Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype |
| --- | --- | --- | --- |
| CUBLAS_COMPUTE_16F or CUBLAS_COMPUTE_16F_PEDANTIC | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F |
| CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_32F_PEDANTIC | CUDA_R_32F | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8I | CUDA_R_32F |
| CUDA_R_16BF | CUDA_R_32F |
| CUDA_R_16F | CUDA_R_32F |
| CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_8I | CUDA_C_32F |
| CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_32F_FAST_16F or CUBLAS_COMPUTE_32F_FAST_16BF or CUBLAS_COMPUTE_32F_FAST_TF32 | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_64F or CUBLAS_COMPUTE_64F_PEDANTIC | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

The `computeType` parameter provided to this function is used only for internal matrix-matrix multiplications.

* * *

#### `cublasMpGemm_bufferSize`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemm-buffersize "Link to this heading")

cublasMpStatus_t cublasMpGemm_bufferSize(
 cublasMpHandle_t handle,
 cublasOperation_t transA,
 cublasOperation_t transB,
 int64_t m,
 int64_t n,
 int64_t k,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 const void* beta,
 void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 cublasComputeType_t computeType,
 size_t* workspaceSizeInBytesOnDevice,
 size_t* workspaceSizeInBytesOnHost);

This function returns the required buffer sizes to perform [cublasMpGemm()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemm-label) on the given input.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| transA | Host | In | Operation op(A) that is non- or (conj.) transpose. |
| transB | Host | In | Operation op(B) that is non- or (conj.) transpose. |
| m | Host | In | Number of rows of sub(A) and sub(C). |
| n | Host | In | Number of columns of sub(B) and sub(C). |
| k | Host | In | Number of columns of sub(A) and rows of sub(B). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| b | Device | In | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). `ib` must be a multiple of the row blocking dimension `mbB`. |
| jb | Host | In | Column index of the first column of the sub(B). `jb` must be a multiple of the column blocking dimension `nbB`. |
| descB | Host | In | Matrix descriptor associated to the global matrix B. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In | Pointer to the first entry of the local portion of the global matrix C. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| computeType | Host | In | [cuBLAS compute type](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t) used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cublasMpGemm()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemm-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cublasMpGemm()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemm-label). |

This routine supports the following combinations of data types:

| Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype |
| --- | --- | --- | --- |
| CUBLAS_COMPUTE_16F or CUBLAS_COMPUTE_16F_PEDANTIC | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F |
| CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_32F_PEDANTIC | CUDA_R_32F | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8I | CUDA_R_32F |
| CUDA_R_16BF | CUDA_R_32F |
| CUDA_R_16F | CUDA_R_32F |
| CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_8I | CUDA_C_32F |
| CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_32F_FAST_16F or CUBLAS_COMPUTE_32F_FAST_16BF or CUBLAS_COMPUTE_32F_FAST_TF32 | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_64F or CUBLAS_COMPUTE_64F_PEDANTIC | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

The `computeType` parameter provided to this function is used only for internal matrix-matrix multiplications.

* * *

### `cublasMpMatmul`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul "Link to this heading")

cublasMpStatus_t cublasMpMatmul(
 cublasMpHandle_t handle,
 cublasMpMatmulDescriptor_t matmulDesc,
 int64_t m,
 int64_t n,
 int64_t k,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 const void* beta,
 const void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 void* d,
 int64_t id,
 int64_t jd,
 cublasMpMatrixDescriptor_t descD,
 void* d_work,
 size_t workspaceSizeInBytesOnDevice,
 void* h_work,
 size_t workspaceSizeInBytesOnHost);

This function performs the matrix-matrix multiplication

_FP8 Support:_

*   FP8 matrix multiplication is only supported for the `TN` format, i.e. , `CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA == CUBLAS_OP_T`, `CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB == CUBLAS_OP_N` on Compute Capability 9.0+ GPUs.

*   To use tensor-scaled FP8 kernels, the following set of requirements must be satisfied:

> *   All matrix dimensions must meet the optimal requirements listed in [Tensor Core Usage](https://docs.nvidia.com/cuda/cublas/#tensor-core-usage) (i.e. pointers and matrix dimension must support 16-byte alignment).
> 
>     *   The compute type must be `CUBLAS_COMPUTE_32F`.
> 
>     *   The scale type must be `CUDA_R_32F`.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| matmulDesc | Host | In | Descriptor of the operation to perform, created with [cublasMpMatmulDescriptorCreate()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmuldescriptorcreate-label). |
| m | Host | In | Number of rows of sub(A) and sub(C). |
| n | Host | In | Number of columns of sub(B) and sub(C). |
| k | Host | In | Number of columns of sub(A) and rows of sub(B). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| b | Device | In | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). `ib` must be a multiple of the row blocking dimension `mbB`. |
| jb | Host | In | Column index of the first column of the sub(B). `jb` must be a multiple of the column blocking dimension `nbB`. |
| descB | Host | In | Matrix descriptor associated to the global matrix B. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In | Pointer to the first entry of the local portion of the global matrix C. `c` can be set to null if `beta == 0`. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| d | Device | Out | Pointer to the first entry of the local portion of the global matrix D. |
| id | Host | In | Row index of the first row of the sub(D). `id` must be a multiple of the row blocking dimension `mbD`. |
| jd | Host | In | Column index of the first column of the sub(D). `jd` must be a multiple of the column blocking dimension `nbD`. |
| descD | Host | In | Matrix descriptor associated to the global matrix D. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cublasMpMatmul_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cublasMpMatmul_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul-buffersize-label). |

This routine supports the following combinations of data types:

| Compute Type | Scale Type (alpha and beta) | Atype | Btype | Ctype | Dtype |
| --- | --- | --- | --- | --- | --- |
| CUBLAS_COMPUTE_16F or CUBLAS_COMPUTE_16F_PEDANTIC | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F |
| CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_32F_PEDANTIC | CUDA_R_32F | CUDA_R_16BF | CUDA_R_16BF | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_16F | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8I | CUDA_R_8I | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_16BF | CUDA_R_16BF | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_16F | CUDA_R_16F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_8I | CUDA_C_8I | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_32F | CUDA_R_32F | CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16BF | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16BF | CUDA_R_8F_E5M2 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16F | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16F | CUDA_R_8F_E5M2 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_8F_E5M2 |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_8F_E5M2 |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_4F_E2M1 | CUDA_R_4F_E2M1 | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_4F_E2M1 | CUDA_R_4F_E2M1 | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_4F_E2M1 | CUDA_R_4F_E2M1 | CUDA_R_32F | CUDA_R_32F |
| CUBLAS_COMPUTE_32F_FAST_16F or CUBLAS_COMPUTE_32F_FAST_16BF or CUBLAS_COMPUTE_32F_FAST_TF32 | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_64F or CUBLAS_COMPUTE_64F_PEDANTIC | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

### NVFP4 requirements[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#nvfp4-requirements "Link to this heading")

*   NVFP4 requires Compute Capability 10.0 and above.

*   Compute Type must be CUBLAS_COMPUTE_32F.

*   Scale Type must be CUDA_R_32F.

*   Scaling mode must be CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3.

The `computeType` parameter provided to this function is used only for internal matrix-matrix multiplications.

* * *

#### `cublasMpMatmul_bufferSize`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul-buffersize "Link to this heading")

cublasMpStatus_t cublasMpMatmul_bufferSize(
 cublasMpHandle_t handle,
 cublasMpMatmulDescriptor_t matmulDesc,
 int64_t m,
 int64_t n,
 int64_t k,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* b,
 int64_t ib,
 int64_t jb,
 cublasMpMatrixDescriptor_t descB,
 const void* beta,
 const void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 void* d,
 int64_t id,
 int64_t jd,
 cublasMpMatrixDescriptor_t descD,
 size_t* workspaceSizeInBytesOnDevice,
 size_t* workspaceSizeInBytesOnHost);

This function returns the required buffer sizes to perform [cublasMpMatmul()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul-label) on the given input.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| matmulDesc | Host | In | Descriptor of the operation to perform, created with [cublasMpMatmulDescriptorCreate()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmuldescriptorcreate-label). |
| m | Host | In | Number of rows of sub(A) and sub(C). |
| n | Host | In | Number of columns of sub(B) and sub(C). |
| k | Host | In | Number of columns of sub(A) and rows of sub(B). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| b | Device | In | Pointer to the first entry of the local portion of the global matrix B. |
| ib | Host | In | Row index of the first row of the sub(B). `ib` must be a multiple of the row blocking dimension `mbB`. |
| jb | Host | In | Column index of the first column of the sub(B). `jb` must be a multiple of the column blocking dimension `nbB`. |
| descB | Host | In | Matrix descriptor associated to the global matrix B. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In | Pointer to the first entry of the local portion of the global matrix C. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| d | Device | Out | Pointer to the first entry of the local portion of the global matrix D. |
| id | Host | In | Row index of the first row of the sub(D). `id` must be a multiple of the row blocking dimension `mbD`. |
| jd | Host | In | Column index of the first column of the sub(D). `jd` must be a multiple of the column blocking dimension `nbD`. |
| descD | Host | In | Matrix descriptor associated to the global matrix D. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cublasMpMatmul()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cublasMpMatmul()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul-label). |

This routine supports the following combinations of data types:

| Compute Type | Scale Type (alpha and beta) | Atype | Btype | Ctype | Dtype |
| --- | --- | --- | --- | --- | --- |
| CUBLAS_COMPUTE_16F or CUBLAS_COMPUTE_16F_PEDANTIC | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F |
| CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_32F_PEDANTIC | CUDA_R_32F | CUDA_R_16BF | CUDA_R_16BF | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_16F | CUDA_R_16F | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8I | CUDA_R_8I | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_16BF | CUDA_R_16BF | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_16F | CUDA_R_16F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_8I | CUDA_C_8I | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_32F | CUDA_R_32F | CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E4M3 | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16BF | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16BF | CUDA_R_8F_E5M2 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16F | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_16F | CUDA_R_8F_E5M2 |
| CUDA_R_8F_E4M3 | CUDA_R_8F_E5M2 | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16BF | CUDA_R_8F_E5M2 |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_8F_E4M3 |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_16F | CUDA_R_8F_E5M2 |
| CUDA_R_8F_E5M2 | CUDA_R_8F_E4M3 | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_4F_E2M1 | CUDA_R_4F_E2M1 | CUDA_R_16BF | CUDA_R_16BF |
| CUDA_R_4F_E2M1 | CUDA_R_4F_E2M1 | CUDA_R_16F | CUDA_R_16F |
| CUDA_R_4F_E2M1 | CUDA_R_4F_E2M1 | CUDA_R_32F | CUDA_R_32F |
| CUBLAS_COMPUTE_32F_FAST_16F or CUBLAS_COMPUTE_32F_FAST_16BF or CUBLAS_COMPUTE_32F_FAST_TF32 | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_64F or CUBLAS_COMPUTE_64F_PEDANTIC | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

### NVFP4 requirements[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#id5 "Link to this heading")

*   NVFP4 requires Compute Capability 10.0 and above.

*   Compute Type must be CUBLAS_COMPUTE_32F.

*   Scale Type must be CUDA_R_32F.

*   Scaling mode must be CUBLASMP_MATMUL_MATRIX_SCALE_VEC16_UE4M3.

The `computeType` parameter provided to this function is used only for internal matrix-matrix multiplications.

* * *

### `cublasMpSyrk`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsyrk "Link to this heading")

cublasMpStatus_t cublasMpSyrk(
 cublasMpHandle_t handle,
 cublasFillMode_t uplo,
 cublasOperation_t trans,
 int64_t n,
 int64_t k,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* beta,
 void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 cublasComputeType_t computeType,
 void* d_work,
 size_t workspaceSizeInBytesOnDevice,
 void* h_work,
 size_t workspaceSizeInBytesOnHost);

This function performs the symmetric rank-  update

where  and  are scalars,  is a symmetric matrix stored in lower or upper mode, and  is a matrix with dimensions  . Also, for matrix

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| uplo | Host | In | Indicates if matrix C lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements. |
| trans | Host | In | Operation op(A) that is non- or transpose. |
| n | Host | In | Number of rows of sub(A) and sub(C). |
| k | Host | In | Number of columns of sub(A). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In/Out | Pointer to the first entry of the local portion of the global matrix C. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| computeType | Host | In | [cuBLAS compute type](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t) used for computations. See table below for supported combinations. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cublasMpSyrk_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsyrk-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cublasMpSyrk_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsyrk-buffersize-label). |

This function requires square block size.

This routine supports the following combinations of data types:

| Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype |
| --- | --- | --- | --- |
| CUBLAS_COMPUTE_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_64F | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

The `computeType` parameter provided to this function is used only for internal matrix-matrix multiplications.

* * *

#### `cublasMpSyrk_bufferSize`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsyrk-buffersize "Link to this heading")

cublasMpStatus_t cublasMpSyrk_bufferSize(
 cublasMpHandle_t handle,
 cublasFillMode_t uplo,
 cublasOperation_t trans,
 int64_t n,
 int64_t k,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* beta,
 void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 cublasComputeType_t computeType,
 size_t* workspaceSizeInBytesOnDevice,
 size_t* workspaceSizeInBytesOnHost);

This function returns the required buffer sizes to perform [cublasMpSyrk()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsyrk-label) on the given input.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| uplo | Host | In | Indicates if matrix C lower or upper part is stored, the other symmetric part is not referenced and is inferred from the stored elements. |
| trans | Host | In | Operation op(A) that is non- or transpose. |
| n | Host | In | Number of rows of sub(A) and sub(C). |
| k | Host | In | Number of columns of sub(A). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In | Pointer to the first entry of the local portion of the global matrix C. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| computeType | Host | In | [cuBLAS compute type](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t) used for computations. See table below for supported combinations. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cublasMpSyrk()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsyrk-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cublasMpSyrk()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsyrk-label). |

This function requires square block size.

This routine supports the following combinations of data types:

| Compute Type | Scale Type (alpha and beta) | Atype/Btype | Ctype |
| --- | --- | --- | --- |
| CUBLAS_COMPUTE_32F | CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUBLAS_COMPUTE_64F | CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

The `computeType` parameter provided to this function is used only for internal matrix-matrix multiplications.

* * *

### `cublasMpGeadd`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgeadd "Link to this heading")

cublasMpStatus_t cublasMpGeadd(
 cublasMpHandle_t handle,
 cublasOperation_t trans,
 int64_t m,
 int64_t n,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* beta,
 void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 void* d_work,
 size_t workspaceSizeInBytesOnDevice,
 void* h_work,
 size_t workspaceSizeInBytesOnHost);

This function performs the matrix-matrix addition

where  and  are scalars, and  and  are matrices stored in column-major format with dimensions  and  , respectively. Also, for matrix

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| trans | Host | In | Operation op(A) that is non- or (conj.) transpose. |
| m | Host | In | Number of rows of sub(A) and sub(C). |
| n | Host | In | Number of columns of sub(A) and sub(C). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In/Out | Pointer to the first entry of the local portion of the global matrix C. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cublasMpGeadd_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgeadd-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cublasMpGeadd_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgeadd-buffersize-label). |

This function requires square block size.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cublasMpGeadd_bufferSize`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgeadd-buffersize "Link to this heading")

cublasMpStatus_t cublasMpGeadd_bufferSize(
 cublasMpHandle_t handle,
 cublasOperation_t trans,
 int64_t m,
 int64_t n,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* beta,
 void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 size_t* workspaceSizeInBytesOnDevice,
 size_t* workspaceSizeInBytesOnHost);

This function returns the required buffer sizes to perform [cublasMpGeadd()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgeadd-label) on the given input.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| trans | Host | In | Operation op(A) that is non- or (conj.) transpose. |
| m | Host | In | Number of rows of sub(A) and sub(C). |
| n | Host | In | Number of columns of sub(A) and sub(C). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In | Pointer to the first entry of the local portion of the global matrix C. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cublasMpGeadd()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgeadd-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cublasMpGeadd()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgeadd-label). |

This function requires square block size.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

### `cublasMpTradd`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd "Link to this heading")

cublasMpStatus_t cublasMpTradd(
 cublasMpHandle_t handle,
 cublasFillMode_t uplo,
 cublasOperation_t trans,
 int64_t m,
 int64_t n,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* beta,
 void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 void* d_work,
 size_t workspaceSizeInBytesOnDevice,
 void* h_work,
 size_t workspaceSizeInBytesOnHost);

This function performs the trapezoidal matrix-matrix addition

where  and  are scalars, and  and  are matrices stored in column-major format with dimensions  and  , respectively. Also, for matrix

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| uplo | Host | In | Indicates if matrix C lower or upper part is stored, the other part is not referenced and is inferred from the stored elements. |
| trans | Host | In | Operation op(A) that is non- or (conj.) transpose. |
| m | Host | In | Number of rows of sub(A) and sub(C). |
| n | Host | In | Number of columns of sub(A) and sub(C). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In/Out | Pointer to the first entry of the local portion of the global matrix C. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| d_work | Device | Out | Device workspace of size `workspaceInBytesOnDevice`. |
| workspaceInBytesOnDevice | Host | In | The size in bytes of the local device workspace needed by the routine as provided by [cublasMpTradd_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd-buffersize-label). |
| h_work | Host | Out | Host workspace of size `workspaceInBytesOnHost`. |
| workspaceInBytesOnHost | Host | In | The size in bytes of the local host workspace needed by the routine as provided by [cublasMpTradd_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd-buffersize-label). |

This function requires square block size.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

* * *

#### `cublasMpTradd_bufferSize`[#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd-buffersize "Link to this heading")

cublasMpStatus_t cublasMpTradd_bufferSize(
 cublasMpHandle_t handle,
 cublasFillMode_t uplo,
 cublasOperation_t trans,
 int64_t m,
 int64_t n,
 const void* alpha,
 const void* a,
 int64_t ia,
 int64_t ja,
 cublasMpMatrixDescriptor_t descA,
 const void* beta,
 void* c,
 int64_t ic,
 int64_t jc,
 cublasMpMatrixDescriptor_t descC,
 size_t* workspaceSizeInBytesOnDevice,
 size_t* workspaceSizeInBytesOnHost);

This function returns the required buffer sizes to perform [cublasMpTradd()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd-label) on the given input.

| Parameter | Memory | In/Out | Description |
| --- | --- | --- | --- |
| handle | Host | In | cuBLASMp library handle. |
| uplo | Host | In | Indicates if matrix C lower or upper part is stored, the other part is not referenced and is inferred from the stored elements. |
| trans | Host | In | Operation op(A) that is non- or (conj.) transpose. |
| m | Host | In | Number of rows of sub(A) and sub(C). |
| n | Host | In | Number of columns of sub(A) and sub(C). |
| alpha | Host | In | <type> scalar used for multiplication. |
| a | Device | In | Pointer to the first entry of the local portion of the global matrix A. |
| ia | Host | In | Row index of the first row of the sub(A). `ia` must be a multiple of the row blocking dimension `mbA`. |
| ja | Host | In | Column index of the first column of the sub(A). `ja` must be a multiple of the column blocking dimension `nbA`. |
| descA | Host | In | Matrix descriptor associated to the global matrix A. |
| beta | Host | In | <type> scalar used for multiplication. |
| c | Device | In | Pointer to the first entry of the local portion of the global matrix C. |
| ic | Host | In | Row index of the first row of the sub(C). `ic` must be a multiple of the row blocking dimension `mbC`. |
| jc | Host | In | Column index of the first column of the sub(C). `jc` must be a multiple of the column blocking dimension `nbC`. |
| descC | Host | In | Matrix descriptor associated to the global matrix C. |
| workspaceInBytesOnDevice | Host | Out | On output, contains the size in bytes of the local device workspace needed by [cublasMpTradd()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd-label). |
| workspaceInBytesOnHost | Host | Out | On output, contains the size in bytes of the local host workspace needed by [cublasMpTradd()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd-label). |

This function requires square block size.

This routine supports the following combinations of data types:

| Data Type of A | computeType | Output Data Type |
| --- | --- | --- |
| CUDA_R_32F | CUDA_R_32F | CUDA_R_32F |
| CUDA_R_64F | CUDA_R_64F | CUDA_R_64F |
| CUDA_C_32F | CUDA_C_32F | CUDA_C_32F |
| CUDA_C_64F | CUDA_C_64F | CUDA_C_64F |

Links/Buttons:
- [#](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd-buffersize)
- [cublasMpHandle_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmphandle-t-label)
- [cublasMpCreate()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpcreate-label)
- [cublasMpStatus_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpstatus-t-label)
- [cublasMpEmulationStrategy_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpemulationstrategy-t-label)
- [grid](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpgrid-t-label)
- [grid layout](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpgridlayout-t-label)
- [cublasMpGridDestroy()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgriddestroy-label)
- [cublasMpMatrixDescriptor_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpmatrixdescriptor-t-label)
- [cublasMpMatmulDescriptor_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpmatmuldescriptor-t-label)
- [cublasMpMatmul()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul-label)
- [cuBLAS compute type](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t)
- [cublasMpMatmulDescriptorAttribute_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmpmatmuldescriptorattribute-t-label)
- [cublasMpGemr2D_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemr2d-buffersize-label)
- [cublasMpGemr2D()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemr2d-label)
- [cublasMpTrmr2D_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrmr2d-buffersize-label)
- [cublasMpTrmr2D()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrmr2d-label)
- [cublasMpLoggerCallback_t](https://docs.nvidia.com/cuda/cublasmp/usage/types.html.md#cublasmploggercallback-t-label)
- [cuBLASMp Logging](https://docs.nvidia.com/cuda/cublasmp/usage/logging.html.md#logging-label)
- [cublasMpTrsm_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrsm-buffersize-label)
- [cublasMpTrsm()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptrsm-label)
- [cublasMpGemm_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemm-buffersize-label)
- [cublasMpGemm()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgemm-label)
- [Tensor Core Usage](https://docs.nvidia.com/cuda/cublas/#tensor-core-usage)
- [cublasMpMatmulDescriptorCreate()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmuldescriptorcreate-label)
- [cublasMpMatmul_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpmatmul-buffersize-label)
- [cublasMpSyrk_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsyrk-buffersize-label)
- [cublasMpSyrk()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpsyrk-label)
- [cublasMpGeadd_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgeadd-buffersize-label)
- [cublasMpGeadd()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmpgeadd-label)
- [cublasMpTradd_bufferSize()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd-buffersize-label)
- [cublasMpTradd()](https://docs.nvidia.com/cuda/cublasmp/usage/functions.html.md#cublasmptradd-label)
