[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

# DALIA (PyINLA)
Python implementation of the methodology of integrated nested Laplace approximations (INLA), putting the accent on portability, modularity and performance.
This project, DALIA, is for now also refered to as PyINLA. A branding change from one to the other is in progress.

If you want to help us in the developement of DALIA, you can fill the following `missing features` survey: https://forms.gle/o4CxBDcr1t73pBHbA

If you want to get involved in the development of DALIA, please feel free to contact us directly.

## Installation
DALIA is a python package that can be installed from its source code. You will need a working `conda` installation as well as the `Serinv` (https://github.com/vincent-maillou/serinv) solver library for accelerated solution of spatio-temporal models.

You can get a working installation of `conda` on the Miniconda website: https://repo.anaconda.com/miniconda/

This package relies on several libraries, some of which enabling high performance computing (HPC) features and GPU acceleration. These libraries (CuPy, MPI4Py, etc.) are not required for the basic functionality of the package, but are required for the advanced features.

Default required packages are:
```bash
conda install numpy scipy
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib tabulate numba -y
```

You can then optionally install the Serinv solver (required for spatio-temporal models)
```bash
cd /path/to/serinv/
python -m pip install -e .
```

And finally install the DALIA package:
```bash
cd /path/to/pyinla/
python -m pip install -e .
```

For more detailed installation instructions, especially on clusters, leveraging GPU acceleration through `CuPy` and distributed computing through `MPI4Py`, please refer to the [dev note](DEV_README.md) in the `DEV_README.md` file.

## Examples
Some examples are provided with running scripts. The examples are being tracked using `git-lfs`, to download them, run the following commands:
```bash
git lfs pull
git lfs checkout
```


## Known Installation Issues
The `sqlite` module might not work properly. Forcing the following version of `sqlite` might help:
```bash
conda install conda-forge::sqlite=3.45.3
```