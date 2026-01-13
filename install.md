# How to install DALIA on one of the default clusters or your personal machine
DALIA has been developed on several (super)computing infrastructures. Since the beginning of the project, our goal has been to provide a seamless experience for users regardless of the underlying hardware and software stack.

Because full generalization is not always possible, we provide a simplified procedure for multiple supercomputing infrastructures (with different hardware) to make installing DALIA on a new cluster as easy as possible.

## Purpose
Base `conda` environments for DALIA are provided in the `dalia/envs` directory. These environments contain all the Python packages on which DALIA relies. Hardware-specific packages (e.g., `cupy` for GPU compute, `mpi4py` for multiprocessing) are not included; they can be installed as extensions of the base environment. We provide base environments for both `x86` and `aarch64` architectures.

These environments are used in the CI/CD pipelines for the respective clusters and are also available for users to quickly set up DALIA on these clusters or any other machine.

## Supported Clusters
We currently use three different clusters for the testing and development of DALIA:

| Organization | Cluster | Arch                              | Description                                                    | Memory                     | Nodes                        |
| ------------ | ------- | --------------------------------- | -------------------------------------------------------------- | -------------------------- | ---------------------------- |
| [FAU](https://doc.nhr.fau.de/)          | [Fritz](https://doc.nhr.fau.de/clusters/fritz/)   | x86 (Sapphire Rapids)             | 2x Intel Xeon Platinum 8470 per node<br>(2x52-cores @ 2.0 GHz) | Up to 2TB DDR5             | 64x 8470, 992x 8360Y         |
| [FAU](https://doc.nhr.fau.de/)          | [Alex](https://doc.nhr.fau.de/clusters/alex/)    | x86 (AMD EPYC 7713)<br>Ampere<br> | 8x Nvidia A100 per node<br>(2x 64-cores CPU + 8 GPUs)          | Up to 80GB HBM2            | 18x A100 80GB, 20x A100 40GB |
| [CSCS](https://www.cscs.ch/)         | [Daint](https://docs.cscs.ch/clusters/daint/)   | ARM (Grace)<br>Hopper             | 4x Nvidia GH200 per node<br>(72-cores CPU + 1 GPU)             | 128GB LPDDR5X<br>96GB HBM3 | 1022                         |


## Environments Configuration Matrix
DALIA is designed to work across a wide variety of hardware and software stacks (e.g., GPU acceleration, distributed memory, vendor-specific libraries). The table below summarizes the supported configurations and the corresponding pre-configured `conda` environments.

|              | No Comm              | Host MPI           | GPU-Aware MPI      | xCCL               |
| :----------- | :------------------- | :----------------- | :----------------- | :----------------- |
| CPU (x86)    | *dalia_base_fritz* | *dalia_hmpi_fritz* | NA                 | NA                 |
| GPU (NVIDIA) | *dalia_base_alex*  | NA  | NA  | *dalia_xccl_alex*  |
| GPU (NVIDIA) | *dalia_base_daint* | NA | NA | *dalia_xccl_daint* |
| CPU (ARM)    | x                    | x                  | x                  | x                  |
| GPU (AMD)    | x                    | x                  | x                  | x                  |

The `dalia_base` environment contains all the dependencies needed to run DALIA on a single node without any communication library (e.g., MPI, xCCL) and without GPU support. This environment includes only hardware-independent Python dependencies. In `dalia/scripts/` we provide interactive installers that can create these `dalia_base` environments for you and, when applicable, extend them to support multi-node communication with MPI or xCCL and/or GPU acceleration.

**Notes:** 
- On the Daint and Alex clusters, CuPy is installed using wheels that include NCCL. Therefore, NCCL is available whenever CuPy is installed. For this reason, no standalone GPU-aware MPI environments are provided.
- On Daint, the `dalia_base_daint` environment allows one to run on the ARM CPU. However, we do not provide any conda environment specifically targeted for a general ARM CPU-cluster.

# Detailed Instructions
## On Fritz@FAU
### a) Installation
1. Clone the repositories to your workspace:    
    ```
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```
2. Source the `fritz_fau_utils.sh` script to access the install utilities:
    ```
    cd DALIA/
    source scripts/fritz_fau_utils.sh
    ```
3. Load the required environment modules:
    ```
    fritz_load_modules
    ```
4. Create the conda environment:
    ```
    fritz_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```
    Notes:
    - This example installs all optional dependencies and uses developer mode. You can also run the installer in interactive mode by simply running `fritz_create_conda_env`.
    - Developer mode (`--dev-mode`) keeps the most performant conda environment available, as well as all environments created along the way. This ensures that during development DALIA can be tested against all supported configurations.
    - The created environment is activated automatically at the end of the installation.
5. Activate the conda environment:
    ```
    fritz_activate_conda_env
    ```
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env="desired_environment"` argument to the function.

### b) Usage
Right after installation, you can use DALIA using the loaded modules and activated conda environment.

However, in general, in future shell sessions you will need to:
1. Source the `fritz_fau_utils.sh` script:
    ```
    source /path/to/DALIA/scripts/fritz_fau_utils.sh
    ```
2. Load the required environment modules:
    ```
    fritz_load_modules
    ```
3. Activate the conda environment:
    ```
    fritz_activate_conda_env
    ```

You will then be ready to use DALIA.

### c) Verify Installation
There is currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite. In an interactive session: `salloc -N 1 --time=00:30:00` (you might need to wait to get the allocation), you can (after activated the correct conda environment) run the test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, you can check the `/path/to/DALIA/examples/run_example_fritz_fau.sh` script for an example on how to submit a job on Fritz.

## On Alex@FAU
### a) Installation

1. Clone the repositories to your workspace:    
    ```
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```
2. Source the `alex_fau_utils.sh` script to access the install utilities:
    ```
    cd DALIA/
    source scripts/alex_fau_utils.sh
    ```
3. Load the required environment modules:
    ```
    alex_load_modules
    ```
4. Create the conda environment:
    ```
    alex_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```
    Notes:
    - This example installs all optional dependencies and uses developer mode. You can also run the installer in interactive mode by simply running `alex_create_conda_env`.
    - Developer mode (`--dev-mode`) keeps the most performant conda environment available, as well as all environments created along the way. This ensures that during development DALIA can be tested against all supported configurations.
    - The created environment is activated automatically at the end of the installation.
5. Activate the conda environment:
    ```
    alex_activate_conda_env
    ```
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument to the function.

### b) Usage
Right after installation, you can use DALIA using the loaded modules and activated conda environment.

However, in general, in future shell sessions you will need to:
1. Source the `alex_fau_utils.sh` script:
    ```
    source /path/to/DALIA/scripts/alex_fau_utils.sh
    ```
2. Load the required environment modules:
    ```
    alex_load_modules
    ```
3. Activate the conda environment:
    ```
    alex_activate_conda_env
    ```

You will then be ready to use DALIA.

### c) Verify Installation
There is currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite. In an interactive session: `salloc --gres=gpu:a100:1 --time=0:30:00` (you might need to wait to get the allocation), you can (after activated the correct conda environment) run the test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, you can check the `/path/to/DALIA/examples/run_example_alex_fau.sh` script for an example on how to submit a job on Alex.


## On Daint@CSCS
### a) Installation
1. Clone the repositories to your workspace:    
    ```
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```

2. Source the `daint_cscs_utils.sh` script to access the install utilities:
    ```
    cd DALIA/
    source scripts/daint_cscs_utils.sh
    ```

3. Use the `daint_install_conda` utility to install Miniconda in your user space (if not already installed):
    ```
    daint_install_conda --yes
    ```
    Notes:
    - This step is only required if you do not already have `conda` installed.
    - The `--yes` option automatically confirms the installation prompts.
    - Additional information about the installer options can be found by running `daint_install_conda --help`.

4. Source the installed conda environment:
    ```
    source ~/miniconda3/etc/profile.d/conda.sh
    ```
    Notes:
    - This is needed only once; in subsequent shell sessions conda will be initialized automatically through your `.bashrc` file.

5. Install the programming environment (`uenv`):
    ```
    daint_install_uenv
    ```

6. Start the programming environment:
    ```
    daint_start_uenv
    ```
    Notes: 
    - This script assumes that you have successfully installed the programming environment in the previous step. If a programming environment is already active, it will be stopped and replaced with a new one.

7. Source the `daint_cscs_utils.sh` script again to access the install utilities:
    ```
    source scripts/daint_cscs_utils.sh
    ```
    Notes:
    - This is needed because starting the programming environment spawns a new shell session in which previously sourced scripts are no longer available.

8. Load the required environment modules:
    ```
    daint_load_modules
    ```

9. Create the conda environment:
    ```
    daint_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```

10. Activate the conda environment:
    ```
    daint_activate_conda_env
    ```
    Note: This function tries to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument.

### b) Usage
...

### c) Verify Installation
...

## On a Personal Machine
### a) Installation
Given a working installation of `git` and `conda`, you can install DALIA on your personal machine as follows:

1. Clone the repositories to your workspace:
    ```
    mv /my/install/path
    git clone https://github.com/dalia-project/DALIA
    git clone https://github.com/vincent-maillou/serinv # Optional, recommended for ST-modeling
    ```

2. Based on your micro-architecture (x86 or aarch64), create the conda environment using the provided installer scripts:
- For x86 architecture:
    ```bash
    conda env create --name "env_name" -f path/to/DALIA/envs/dalia_base_x86.yml
    ```
- For aarch64 architecture:
    ```bash
    conda env create --name "env_name" -f path/to/DALIA/envs/dalia_base_aarch64.yml
    ```

3. Activate the conda environment:
    ```bash
    conda activate env_name
    ```

4. Install DALIA in editable mode:
    ```bash
    cd path/to/DALIA
    pip install -e .
    ```

5. Install Serinv (optional, recommended for ST-modeling) in editable mode:
    ```bash
    cd path/to/serinv
    pip install -e .
    ```

### b) Verify Installation
They are currently two ways to verify that DALIA has been installed correctly:
1. Run the provided test suite: `./path/to/DALIA/tests/runner.sh`.
2. Run one of the provided examples, for example: `python /path/to/DALIA/examples/gr/run.py`


## Notes on `git-lfs`
In order to handle large files (e.g. examples datasets), we use `git-lfs` (Git Large File Storage). Some files are then tracked by `git-lfs` instead of `git`, and are only downloaded on demand.

In order to run the examples provided with DALIA, you need to have `git-lfs` installed and configured on your system.

On Fritz and Alex, `git-lfs` is available by default and is in your path: `/usr/bin/git-lfs`. However on Daint this is not the case, and `git-lfs` is not provided by the system. We provide a utility to install `git-lfs` in your user space as part of the `daint_cscs_utils.sh` script. 

After sourcing the `daint_cscs_utils.sh` script, you can run the following command to install `git-lfs`:
```
daint_install_git_lfs
```

## Other Informations

- After installing packages with conda, it is recommended to run `conda clean --all` to free up disk space.
- These installation procedures and `conda` environments have been tested on Linux systems based on both `x86` and `aarch64` architectures.
- If the `sqlite` module does not work properly, forcing the following version might help:
    ```bash
    conda install conda-forge::sqlite=3.45.3
    ```
- CuPy switched NCCL to lazy import, which means NCCL must be imported explicitly with `from cupy.cuda import nccl` before it is available in `cupy.cuda.nccl`.
