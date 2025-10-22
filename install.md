# How to install DALIA on one of the default clusters
DALIA have been developped on several (super)computing infrastructures, since the beginning of the project it has been our goal to provide a seamless experience for users, regardless of the underlying hardware and software stack.

As this generalization is not always possible, we provide a simplified procedure for several supercomputing infrastructure (made of different hardware), with the goal of rendering the installation experience of DALIA on a new cluster as easy as possible.

## Purpose
A base `conda` environment for DALIA is provided in the `dalia/envs` directory. This environment contains all the Python packages on wich DALIA relies. Are not included in this environment the hardware-specific packages (e.g. CuPy for GPU-compute), these packages can be installed as extensions of the provided base environment.

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
| CPU (x86)    | ==dalia_base== | *dalia_hmpi_fritz* | NA                 | NA                 |
| GPU (NVIDIA) | ==dalia_base==  | *dalia_hmpi_alex*  | *dalia_ampi_alex*  | *dalia_xccl_alex*  |
| GPU (NVIDIA) | ==dalia_base== | *dalia_hmpi_daint* | *dalia_ampi_daint* | *dalia_xccl_daint* |
| CPU (ARM)    | x                    | x                  | x                  | x                  |
| GPU (AMD)    | x                    | x                  | x                  | x                  |

The `dalia_base` environment contains all the necessary dependencies to run DALIA on a single node without any communication library (e.g. MPI, xCCL) and without GPU support. This environment only contains hardware-independent python dependencies. We provide in `dalia/scripts/` interactive installer that not only can create this `dalia_base` environments for you, but also extend it to support, when applicable, multi-node communication with MPI or xCCL and/or GPU acceleration.


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
    - The created environment will be activated automatically at the end of the installation.
5. Activate the conda environment:
    ```
    fritz_activate_conda_env
    ```
    Note: This function will try to activate the most performant environment available on the cluster. You can also activate a specific environment by providing the `--env` argument to the function.


## On Alex@FAU
... todo

## On Daint@CSCS
... todo