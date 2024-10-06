#!/bin/bash -l
#SBATCH --job-name=test_pyinla
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --account=hck
#SBATCH --reservation=eurohack24

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### make it run pytest from the pyINLA base directory
cd $HOME/pyINLA

srun pytest .