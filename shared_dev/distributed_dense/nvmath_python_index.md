Title: nvmath-python: Unleashing the Full Capabilities of NVIDIA Math Libraries within Python#

URL Source: https://docs.nvidia.com/cuda/nvmath-python/latest/index.html

Published Time: Mon, 08 Dec 2025 19:25:15 GMT

Markdown Content:
Welcome to the nvmath-python documentation!

**nvmath-python** is a Python library to enable cutting edge performance, productivity, and interoperability within the Python computational ecosystem through NVIDIA’s high-performance math libraries.

To quickly get started with nvmath-python, take a look at our [Getting Started](https://docs.nvidia.com/cuda/nvmath-python/latest/quickstart.html.md) manual. Refer to our [Installation Guide](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html.md) for detailed instructions on the various installation choices available.

Contents[#](https://docs.nvidia.com/cuda/nvmath-python/latest/index.html.md#contents "Link to this heading")
---------------------------------------------------------------------------------------------------------

User Guide

*   [Getting Started](https://docs.nvidia.com/cuda/nvmath-python/latest/quickstart.html.md)
    *   [Installation](https://docs.nvidia.com/cuda/nvmath-python/latest/quickstart.html.md#installation)
    *   [Examples](https://docs.nvidia.com/cuda/nvmath-python/latest/quickstart.html.md#examples)

*   [Overview](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html.md)
    *   [Architecture](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html.md#architecture)
    *   [Host APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html.md#host-apis)
    *   [Host APIs with Callbacks](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html.md#host-apis-with-callbacks)
    *   [Device APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html.md#device-apis)
    *   [Compatibility Policy](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html.md#compatibility-policy)

*   [Installation](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html.md)
    *   [Install nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html.md#install-nvmath-python)
    *   [Run nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html.md#run-nvmath-python)
    *   [Troubleshooting](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html.md#troubleshooting)

Examples and tutorials

*   [Linear Algebra Host APIs Tutorial](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/linalg.html.md)
    *   [Introduction to GEMM with nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/notebooks/matmul/01_introduction.html.md)
    *   [Fused Epilogs](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/notebooks/matmul/02_epilogs.html.md)
    *   [Implementing a simple neural network](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/notebooks/matmul/03_backpropagation.html.md)
    *   [Narrow-precision operations](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/notebooks/matmul/04_fp8.html.md)

*   [Examples on GitHub](https://github.com/NVIDIA/nvmath-python/tree/main/examples)

API Reference

*   [Host APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/index.html.md)
    *   [Key Concepts](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/index.html.md#key-concepts)
        *   [Matrix and Tensor Qualifiers](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/index.html.md#matrix-and-tensor-qualifiers)

    *   [Contents](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/index.html.md#contents)
        *   [Linear Algebra](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/linalg/index.html.md)
        *   [Sparse Linear Algebra](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/index.html.md)
        *   [Fast Fourier Transform](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/fft/index.html.md)
        *   [Tensor Operations](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/tensor/index.html.md)
        *   [Host API Utilities](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/utils.html.md)

*   [Device APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/index.html.md)
    *   [Device API utilities](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/utils.html.md)
        *   [Overview](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/utils.html.md#overview)
        *   [API Reference](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/utils.html.md#api-reference)

    *   [cuBLASDx](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/cublas.html.md)
        *   [Overview](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/cublas.html.md#overview)
        *   [API Reference](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/cublas.html.md#api-reference)

    *   [cuFFTDx](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/cufft.html.md)
        *   [Overview](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/cufft.html.md#overview)
        *   [API Reference](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/cufft.html.md#api-reference)

    *   [cuRAND Device APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/curand.html.md)
        *   [Overview](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/curand.html.md#overview)
        *   [API Reference](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/curand.html.md#module-nvmath.device.random)

*   [Distributed APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html.md)
    *   [Overview](https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html.md#overview)
        *   [Contents](https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html.md#contents)

*   [Bindings](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md)
    *   [Overview](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#overview)
    *   [Naming & Calling Convention](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#naming-calling-convention)
    *   [Memory management](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#memory-management)
        *   [Pointer and data lifetime](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#pointer-and-data-lifetime)

    *   [API Reference](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#api-reference)
        *   [cuBLAS (`nvmath.bindings.cublas`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cublas.html.md)
        *   [cuBLASLt (`nvmath.bindings.cublaslt`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cublasLt.html.md)
        *   [cuBLASMp (`nvmath.bindings.cublasMp`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cublasMp.html.md)
        *   [cuDSS (`nvmath.bindings.cudss`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cudss.html.md)
        *   [cuFFT (`nvmath.bindings.cufft`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cufft.html.md)
        *   [cuSOLVER (`nvmath.bindings.cusolver`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cusolver.html.md)
        *   [cuSOLVERDn (`nvmath.bindings.cusolverDn`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cusolverDn.html.md)
        *   [cuSPARSE (`nvmath.bindings.cusparse`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cusparse.html.md)
        *   [cuRAND (`nvmath.bindings.curand`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/curand.html.md)
        *   [NVPL BLAS (`nvmath.bindings.nvpl.blas`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/nvpl.blas.html.md)
        *   [NVPL FFT (`nvmath.bindings.nvpl.fft`)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/nvpl.fft.html.md)

Links/Buttons:
- [#](https://docs.nvidia.com/cuda/nvmath-python/latest/index.html.md#contents)
- [Getting Started](https://docs.nvidia.com/cuda/nvmath-python/latest/quickstart.html.md)
- [Installation Guide](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html.md)
- [Installation](https://docs.nvidia.com/cuda/nvmath-python/latest/quickstart.html.md#installation)
- [Examples](https://docs.nvidia.com/cuda/nvmath-python/latest/quickstart.html.md#examples)
- [Overview](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#overview)
- [Architecture](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html.md#architecture)
- [Host APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/index.html.md)
- [Host APIs with Callbacks](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html.md#host-apis-with-callbacks)
- [Device APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/index.html.md)
- [Compatibility Policy](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html.md#compatibility-policy)
- [Install nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html.md#install-nvmath-python)
- [Run nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html.md#run-nvmath-python)
- [Troubleshooting](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html.md#troubleshooting)
- [Linear Algebra Host APIs Tutorial](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/linalg.html.md)
- [Introduction to GEMM with nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/notebooks/matmul/01_introduction.html.md)
- [Fused Epilogs](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/notebooks/matmul/02_epilogs.html.md)
- [Implementing a simple neural network](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/notebooks/matmul/03_backpropagation.html.md)
- [Narrow-precision operations](https://docs.nvidia.com/cuda/nvmath-python/latest/tutorials/notebooks/matmul/04_fp8.html.md)
- [Examples on GitHub](https://github.com/NVIDIA/nvmath-python/tree/main/examples)
- [Key Concepts](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/index.html.md#key-concepts)
- [Matrix and Tensor Qualifiers](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/index.html.md#matrix-and-tensor-qualifiers)
- [Contents](https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html.md#contents)
- [Linear Algebra](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/linalg/index.html.md)
- [Sparse Linear Algebra](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/index.html.md)
- [Fast Fourier Transform](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/fft/index.html.md)
- [Tensor Operations](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/tensor/index.html.md)
- [Host API Utilities](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/utils.html.md)
- [Device API utilities](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/utils.html.md)
- [API Reference](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#api-reference)
- [cuBLASDx](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/cublas.html.md)
- [cuFFTDx](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/cufft.html.md)
- [cuRAND Device APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/curand.html.md)
- [Distributed APIs](https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html.md)
- [Bindings](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md)
- [Naming & Calling Convention](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#naming-calling-convention)
- [Memory management](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#memory-management)
- [Pointer and data lifetime](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html.md#pointer-and-data-lifetime)
- [cuBLAS (nvmath.bindings.cublas)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cublas.html.md)
- [cuBLASLt (nvmath.bindings.cublaslt)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cublasLt.html.md)
- [cuBLASMp (nvmath.bindings.cublasMp)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cublasMp.html.md)
- [cuDSS (nvmath.bindings.cudss)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cudss.html.md)
- [cuFFT (nvmath.bindings.cufft)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cufft.html.md)
- [cuSOLVER (nvmath.bindings.cusolver)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cusolver.html.md)
- [cuSOLVERDn (nvmath.bindings.cusolverDn)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cusolverDn.html.md)
- [cuSPARSE (nvmath.bindings.cusparse)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/cusparse.html.md)
- [cuRAND (nvmath.bindings.curand)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/curand.html.md)
- [NVPL BLAS (nvmath.bindings.nvpl.blas)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/nvpl.blas.html.md)
- [NVPL FFT (nvmath.bindings.nvpl.fft)](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/nvpl.fft.html.md)
- [Release Notes](https://docs.nvidia.com/cuda/nvmath-python/latest/release-notes.html.md)
- [Code of Conduct](https://docs.nvidia.com/cuda/nvmath-python/latest/CODE_OF_CONDUCT.html.md)
- [Contributing](https://docs.nvidia.com/cuda/nvmath-python/latest/CONTRIBUTING.html.md)
- [License](https://docs.nvidia.com/cuda/nvmath-python/latest/license.html.md)
- [GitHub Repository](https://github.com/NVIDIA/nvmath-python)
