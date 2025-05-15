#!/bin/bash -l
#SBATCH --job-name="examples"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=sm96
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
####SBATCH --partition=normal
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules

set -e -u

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1

export NCCL_NET='AWS Libfabric'
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

source ~/load_modules.sh
conda activate allin

export ARRAY_MODULE=cupy
export MPI_CUDA_AWARE=0

# --- How to Run ---
# This run script is designed to run on the Daint supercomputer at CSCS.
# It uses SLURM for job scheduling and assumes that the user has a working 
# installation of pyINLA and its dependencies. By default, PyINLA will exploit  
# job parallelism at the parallel function evaluation level.
# --- Parameters ---
# `--solver_min_p` : The minimum number of Processes(/GPUs) to use for the structured 
#                    solver. The default is 1. The maximum number of processes is
# `--max_iter` : The maximum number of iterations of the minimization.

# --- Run Spatio-temporal Examples ---
# srun python /users/vmaillou/repositories/pyINLA/examples/gst_small/run.py --solver_min_p 1
srun python /users/vmaillou/repositories/pyINLA/examples/gst_medium/run.py --solver_min_p 1 --max_iter 1
# srun python /users/vmaillou/repositories/pyINLA/examples/gst_large/run.py --solver_min_p 1

# --- Run Coregional (Spatial) Examples ---
# srun python /users/vmaillou/repositories/pyINLA/examples/gs_coreg2_small/run.py
# srun python /users/vmaillou/repositories/pyINLA/examples/gs_coreg3_small/run.py

# --- Run Coregional (Spatio-temporal) Examples ---
# srun python /users/vmaillou/repositories/pyINLA/examples/gst_coreg2_small/run.py --solver_min_p 1
# srun python /users/vmaillou/repositories/pyINLA/examples/gst_coreg3_small/run.py --solver_min_p 1 --max_iter 1

