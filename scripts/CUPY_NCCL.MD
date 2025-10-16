# ALPS: CuPy and NCCL Installation Instructions
To install NCCL you first need to have all the correct modules and env_path set:
MODULES:
```bash
module load cuda gcc meson ninja nccl cray-mpich cmake openblas aws-ofi-nccl netlib-scalapack
```

PATHS:
```bash
# NCCL environment setup
export NCCL_ROOT=/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd
export NCCL_LIB_DIR=$NCCL_ROOT/lib
export NCCL_INCLUDE_DIR=$NCCL_ROOT/include

export CUDA_DIR=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$NCCL_ROOT/include:$CPATH
export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH
```

Then install cupy with the following command:
```bash
pip install --no-cache-dir cupy-cuda12x
```