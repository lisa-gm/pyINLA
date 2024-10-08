#!/bin/bash -l
#SBATCH --job-name=bench_spatio_temporal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --reservation=eurohack24

set -x
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate pyinla

srun -n 1 python run.py