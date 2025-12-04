# How to install DALIA on one of the default clusters
DALIA have been developped on several (super)computing infrastructures, since the beginning of the project it has been our goal to provide a seamless experience for users, regardless of the underlying hardware and software stack.

As this generalization is not always possible, we provide a simplified procedure for several supercomputing infrastructure (made of different hardware), with the goal of rendering the installation experience of DALIA on a new cluster as easy as possible.

## Purpose
Base `conda` environments for DALIA are provided in the `dalia/envs` directory. These environments contains all the Python packages on wich DALIA relies. Are not included in these environments the hardware-specific packages (e.g. `cupy` for GPU-compute, `mpi4py` for multiprocessing capabilities), these packages can be installed as extensions of the provided base environment. We provide base environment for both `x86` and `aarch64` architectures. 

This environment is being used for the CI/CD pipelines of DALIA on the respective clusters and is also available for users to quickly setup DALIA on these clusters or any other machine.

## Supported Clusters
We currently use three different clusters for the testing and development of DALIA:

| Organization | Cluster | Arch                              | Description                                                    | Memory                     | Nodes                        |
| ------------ | ------- | --------------------------------- | -------------------------------------------------------------- | -------------------------- | ---------------------------- |
| [FAU](https://doc.nhr.fau.de/)          | [Fritz](https://doc.nhr.fau.de/clusters/fritz/)   | x86 (Sapphire Rapids)             | 2x Intel Xeon Platinum 8470 per node<br>(2x52-cores @ 2.0 GHz) | Up to 2TB DDR5             | 64x 8470, 992x 8360Y         |
| [FAU](https://doc.nhr.fau.de/)          | [Alex](https://doc.nhr.fau.de/clusters/alex/)    | x86 (AMD EPYC 7713)<br>Ampere<br> | 8x Nvidia A100 per node<br>(2x 64-cores CPU + 8 GPUs)          | Up to 80GB HBM2            | 18x A100 80GB, 20x A100 40GB |
| [CSCS](https://www.cscs.ch/)         | [Daint](https://docs.cscs.ch/clusters/daint/)   | ARM (Grace)<br>Hopper             | 4x Nvidia GH200 per node<br>(72-cores CPU + 1 GPU)             | 128GB LPDDR5X<br>96GB HBM3 | 1022                         |


## Environments Configuration Matrix
DALIA is supposed to work across a wide variety of hardware and software stacks e.g. GPU accelerated, distributed memory, vendor specific libraries, etc. The following table summarizes the different configurations that are supported by DALIA and for which pre-configured `conda` environments are provided.

|              | No Comm              | Host MPI           | GPU-Aware MPI      | xCCL               |
| :----------- | :------------------- | :----------------- | :----------------- | :----------------- |
| CPU (x86)    | *dalia_base_fritz* | *dalia_hmpi_fritz* | NA                 | NA                 |
| GPU (NVIDIA) | *dalia_base_alex*  | NA  | *dalia_ampi_alex*  | *dalia_xccl_alex*  |
| GPU (NVIDIA) | *dalia_base_daint* | NA | NA | *dalia_xccl_daint* |
| CPU (ARM)    | x                    | x                  | x                  | x                  |
| GPU (AMD)    | x                    | x                  | x                  | x                  |

The `dalia_base` environment contains all the necessary dependencies to run DALIA on a single node without any communication library (e.g. MPI, xCCL) and without GPU support. This environment only contains hardware-independent python dependencies. We provide in `dalia/scripts/` interactive installer that not only can create this `dalia_base` environments for you, but also extend it to support, when applicable, multi-node communication with MPI or xCCL and/or GPU acceleration.

**Notes:** On the Alex and Daint clusters, CuPy is installed using Wheels which comes with NCCL pre-installed. Therefore, NCCL is available whenever CuPy is installed on these clusters. For this reason, they are no GPU-Aware MPI environments alone provided. 

# Detailed Instructions
## On Fritz@FAU
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
3. Load the required environments modules:
    ```
    fritz_load_modules
    ```
4. Create the conda environment:
    ```
    fritz_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --dev-mode
    ```
    Notes:
    - Here a complete installation, with all optional dependencies and developper mode, is provided. You can also run the installer in interactive mode by simply running `fritz_create_conda_env`.    
    - The developer mode `--dev-mode` will not only keep the most performant conda environment available, but also all the conda environments created along the way. This ensure that during developement DALIA can be tested against all possible configurations.
    - The created environment will be activated automatically at the end of the installation.
5. Activate the conda environment:
    ```
    fritz_activate_conda_env
    ```
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument to the function.


## On Alex@FAU
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
3. Load the required environments modules:
    ```
    alex_load_modules
    ```
4. Create the conda environment:
    ```
    alex_create_conda_env --dalia-path=path/to/DALIA --serinv-path=path/to/serinv --install-mpi4py --install-nccl --dev-mode
    ```
    Notes:
    - Here a complete installation, with all optional dependencies and developper mode, is provided. You can also run the installer in interactive mode by simply running `alex_create_conda_env`.
    - The developer mode `--dev-mode` will not only keep the most performant conda environment available, but also all the conda environments created along the way. This ensure that during developement DALIA can be tested against all possible configurations.
    - The created environment will be activated automatically at the end of the installation.
5. Activate the conda environment:
    ```
    alex_activate_conda_env
    ```
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument to the function.

## On Daint@CSCS
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
    - This step is only required if you do not have `conda` installed already.
    - The `--yes` option automatically confirms the installation prompts.
    - Additional informations about the installer options can be found by running `daint_install_conda --help`.

4. Source the installed conda environment:
    ```
    source ~/miniconda3/etc/profile.d/conda.sh
    ```
    Notes:
    - This is only needed this time, in any subsequent shell sessions conda will be initialized automatically through your `.bashrc` file.

5. Install the programming environment (`uenv`):
    ```
    daint_install_uenv
    ```

6. Start the programming environment:
    ```
    daint_start_uenv
    ```

7. Source the `daint_cscs_utils.sh` script again to access the install utilities:
    ```
    source scripts/daint_cscs_utils.sh
    ```
    Notes:
    - This is needed because starting the programming environment spawns a new shell session, in which previously sourced scripts are not available anymore.

8. Load the required environments modules:
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
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument to the function.


## Others Informations

- After using conda for installing packages, it is recommended to run a cleanup using `conda clean --all`.
- These installation procedures and `conda` environments have been tested on Linux systems based on both `x86` and `aarch64` architectures.
- The `sqlite` module might not work properly, in this case forcing the following version of `sqlite` might help:
    ```bash
    conda install conda-forge::sqlite=3.45.3
    ```
- It is worth mentionning that CuPy switched NCCL to lazy import, which means that NCCL needs to be imported first explictly using `from cupy.cuda import nccl` before being available in `cupy.cuda.nccl`.