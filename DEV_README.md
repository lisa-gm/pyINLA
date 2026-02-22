## CSCS @ daint.alps cluster
Here are some installation guidelines to install the project on the Piz Daint cluster of the ALPS supercomputer.
1. Pull and start the necessary `uenv`:
```bash
uenv image find
uenv repo create
uenv image pull prgenv-gnu/24.11:v1
uenv start --view=modules prgenv-gnu/24.11:v1
```
2. Load the necessary modules:
```bash
module load cuda
module load gcc
module load meson
module load ninja
module load nccl
module load cray-mpich
module load cmake
module load openblas
module load aws-ofi-nccl
```
3. Export library PATH:
```bash
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
4. Install miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
chmod u+x Miniconda3-latest-Linux-aarch64.sh
./Miniconda3-latest-Linux-aarch64.sh
```
5. Create the conda environment and install the required libraries:
```bash
conda create -n myenv
conda activate myenv
conda install python=3.12
conda install numpy scipy
MPICC=$(which mpicc) python -m pip install --no-cache-dir mpi4py
pip install cupy --no-dependencies --no-cache-dir
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib tabulate numba -y
# Test the NCCL/CuPy installation
python -c "from cupy.cuda.nccl import *"
```
6. (Optional) Install serinv and run the tests:
```bash
git clone https://github.com/vincent-maillou/serinv # https://github.com/vincent-maillou/serinv/tree/dev
cd /path/to/serinv/
python -m pip install -e .
# Run the sequential tests.
pytest .
```
7. Install dalia
```bash
cd /path/to/dalia/
python -m pip install -e .
```

### Known Installation Issues
The `sqlite` module might not work properly. Forcing the following version of `sqlite` might help:
```bash
conda install conda-forge::sqlite=3.45.3
```

## Erlangen @ Fau cluster
Here are some installation guidelines to install the project on the Fau; Alex and Fritz clusters.
We recommend to test any development in 3 separated environments:
- Bare: The environment without any MPI or GPU support
- Fritz: The environment with MPI support, CPU backend
- Alex: The environment with MPI support, GPU backend (optional: NCCL)

This ensure compatibility no matter the available backend.

```bash
# --- Alex-env ---
module load python
module load openmpi/4.1.6-nvhpc23.7-cuda
module load cuda/12.6.1

conda create -n alex
conda activate alex

CFLAGS=-noswitcherror MPICC=$(which mpicc) pip install --no-cache-dir mpi4py

salloc --partition=a40 --nodes=1 --gres=gpu:a40:1 --time 01:00:00
conda activate alex

conda install -c conda-forge cupy-core
conda install blas=*=*mkl
conda install libblas=*=*mkl
conda install numpy scipy
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib tabulate numba -y

cd /path/to/serinv/
python -m pip install -e .

cd /path/to/dalia/
python -m pip install -e .
```

```bash
# --- Fritz-env ---
module load python
module load openmpi/4.1.2-gcc11.2.0

conda create -n fritz
conda activate fritz

MPICC=$(which mpicc) pip install --no-cache-dir mpi4py

salloc -N 4 --time 01:00:00
conda activate fritz

conda install blas=*=*mkl
conda install libblas=*=*mkl
conda install numpy scipy
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib tabulate numba -y

cd /path/to/serinv/
python -m pip install -e .

cd /path/to/dalia/
python -m pip install -e .
```

```bash
# --- Bare-env ---
module load python
conda create -n bare

salloc -N 4 --time 01:00:00
conda activate bare

conda install blas=*=*mkl
conda install libblas=*=*mkl
conda install numpy scipy
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib tabulate numba -y

cd /path/to/serinv/
python -m pip install -e .

cd /path/to/dalia/
python -m pip install -e .
```

