#!/bin/bash
#####SBATCH --uenv=prgenv-gnu/24.7:v3
#####SBATCH --view=modules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
####SBATCH --gpus-per-task=1
#SBATCH --output=output_gaussian_spatioTemporal_medSize.out
#SBATCH --error=output_gaussian_spatioTemporal_medSize.err
#####SBATCH -C gpu
#SBATCH --reservation=eurohack24
####SBATCH --partition=debug
#SBATCH --time=00:15:00

num_ranks=1

#set -x

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$((OMP_NUM_THREADS-1))

#conda init
#conda deactivate 

#source ~/start_uenv.sh
#source ~/load_modules.sh

 . "/users/hck24/miniconda3/etc/profile.d/conda.sh"

#conda init
conda activate pyinla


export NSYS=1
export NSYS_FILE=gaussian_spatioTemporal_medSize_pinned_numaCtl_omp${SLURM_CPUS_PER_TASK}_numRanks${num_ranks}

# export OMP_PROC_BIND=true
# export OMP_PLACES=cores

srun -n ${num_ranks} numactl -l --all --physcpubind=0-${SLURM_CPUS_PER_TASK} ./nsys.sh python sandbox_gaussian_spatioTemporal_medSize.py >output_gaussian_spatioTemporal_medSize_pinned_numaCtl_omp${SLURM_CPUS_PER_TASK}_numRanks${num_ranks}.txt
