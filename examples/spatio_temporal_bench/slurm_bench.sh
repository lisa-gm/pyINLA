#!/bin/bash -l
#SBATCH --job-name=bench_spatio_temporal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
####SBATCH --account=hck23
#SBATCH --reservation=eurohack24

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate pyinla

srun python run.py