#!/bin/bash -l
#SBATCH --job-name="dalia_alps"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=sm96
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules

# Set DALIA environment variables for examples  
source ../scripts/alps_cscs_utils.sh && alps_load_modules && alps_activate_conda_env && alps_set_perfenv
source ./scripts/dalia_job_utils.sh && dalia_set_perfenv && dalia_print_job_config

# Change to examples directory
if [[ "$(basename "$(pwd)")" != "examples" ]]; then
    echo "❌ Error: Not in examples directory"
    echo "   Current directory: $(pwd)"
    echo "   Please run this script from the examples/ directory"
    exit 1
fi

# --- How to Run ---
# This run script is designed to run on the Daint supercomputer at CSCS.
# It uses SLURM for job scheduling and assumes that the user has a working 
# installation of DALIA and its dependencies. By default, DALIA will exploit  
# job parallelism in a cascade, first at the function evaluation level,
# then at the precision matrix level, finally at the structured solver level.

# --- Parameters ---
# `--solver_min_p` : The minimum number of Processes(/GPUs) to use for the structured 
#                    solver. The default is 1. The maximum number of processes is
# `--max_iter` : The maximum number of iterations of the minimization.

# --- Run Regression Example ---
echo "Regression Example..."
srun python ./regression/run.py --max_iter 100

# --- Run Spatial Examples ---
#echo "Spatial Example (small)..."
#srun python ./gs_small/run.py --max_iter 100

# --- Run Spatio-temporal Examples ---
# echo "Spatio-temporal Example (small)..."
# srun python ./gst_small/run.py --solver_min_p 1 --max_iter 100

# echo "Spatio-temporal Example (medium)..."
# srun python ./gst_medium/run.py --solver_min_p 1 --max_iter 100

#echo "Spatio-temporal Example (large)..."
#srun python ./gst_large/run.py --solver_min_p 1 --max_iter 100

# --- Run Coregional (Spatial) Examples ---
#echo "Coregional Spatial Example (2 models)..."
#srun python ./gs_coreg2_small/run.py --max_iter 100

#echo "Coregional Spatial Example (3 models)..."
#srun python ./gs_coreg3_small/run.py --max_iter 100

# --- Run Coregional (Spatio-temporal) Examples ---
#echo "Coregional Spatio-temporal Example (2 models)..."
#srun python ./gst_coreg2_small/run.py --solver_min_p 1 --max_iter 100

#echo "Coregional Spatio-temporal Example (3 models)..."
#srun python ./gst_coreg3_small/run.py --solver_min_p 1 --max_iter 100
