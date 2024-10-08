# pyINLA
Python implementation of the methodology of integrated nested Laplace approximations (INLA)


## How to install
```
1. Create a conda environment from the environment.yml file
    $ conda env create -f environment.yml
    $ conda activate pyinla

2. Install mpi4py : (Optional)
    $ python -m pip cache remove mpi4py
    $ python -m pip install --no-cache-dir mpi4py

    Please refere to https://mpi4py.readthedocs.io/en/stable/install.html for more details.

3. Install CuPy : (Optional)
    # conda install -c conda-forge cupy cuda-version=xx.x cupy-version=xx.x

    Please refere to https://docs.cupy.dev/en/stable/install.html for more details.

4. Install SerinV : (Optional)
    $ cd path/to/install/folder
    $ git clone https://github.com/vincent-maillou/serinv
    $ cd serinv
    $ pip install --no-dependencies -e .

5. Install pyINLA
    $ cd path/to/pyinla
    $ pip install --no-dependencies -e .
```

If by any chances you run on a weird Apple Mac with arm64 architecture, ```conda-forge``` packages might not be available. In this case, you can install the dependencies manually using ```pip```:
```
$ conda create -n pyinla python=3.11
$ conda activate pyinla
$ pip install numpy scipy matplotlib pydantic pytest pytest-cov pytest-mpi coverage black isort ruff pre-commit
$ cd path/to/pyinla
$ pip install --no-dependencies -e .
```

Conda-forge dependancies
```
conda install -c conda-forge numpy scipy matplotlib pydantic pytest pytest-cov pytest-mpi coverage black isort ruff pre-commit
```