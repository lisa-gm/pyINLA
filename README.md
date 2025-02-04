# pyINLA
Python implementation of the methodology of integrated nested Laplace approximations (INLA)

## Dev-note
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
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib numba -y

cd /path/to/serinv/
python -m pip install -e .

cd /path/to/pyinla/
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
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib numba -y

cd /path/to/serinv/
python -m pip install -e .

cd /path/to/pyinla/
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
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib numba -y

cd /path/to/serinv/
python -m pip install -e .

cd /path/to/pyinla/
python -m pip install -e .
```