#!/bin/bash
#####SBATCH --uenv=prgenv-gnu/24.7:v3
#########SBATCH --view=modules
#SBATCH --ntasks=1
####SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=output_gaussian_regression.out
#SBATCH --error=output_gaussian_regression.err
#SBATCH -C gpu
#SBATCH --partition=debug
#SBATCH --time=00:05:00

export OMP_NUM_THREADS=32

num_ranks=1

#conda init
#conda deactivate 

#source ~/start_uenv.sh
#source ~/load_modules.sh

 . "/users/hck24/miniconda3/etc/profile.d/conda.sh"

#conda init
conda activate pyinla

srun -n ${num_ranks} python sandbox_gaussian_regression.py >output_gaussian_regression_numRanks${num_ranks}.txt
