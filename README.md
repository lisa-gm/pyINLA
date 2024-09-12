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