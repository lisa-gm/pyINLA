#!/bin/bash
#####SBATCH --uenv=prgenv-gnu/24.7:v3
#####SBATCH --view=modules
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --output=output_gaussian_spatioTemporal_medSize.out
#SBATCH --error=output_gaussian_spatioTemporal_medSize.err
###SBATCH -C gpu
#SBATCH --reservation=eurohack24
####SBATCH --partition=debug
#SBATCH --time=00:15:00

#export CUDA_VISIBLE_DEVICES=


num_ranks=${SLURM_NTASKS}

#set -x

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=$((OMP_NUM_THREADS-1))

#conda init
#conda deactivate 

#source ~/start_uenv.sh
#source ~/load_modules.sh

 . "/users/hck24/miniconda3/etc/profile.d/conda.sh"

#conda init
conda activate pyinla

# omp${OMP_NUM_THREADS}
export NSYS=1
export NSYS_FILE=gaussian_spatioTemporal_medSize_pinned_numaCtl_gpu_numRanks${num_ranks}

# export OMP_PROC_BIND=true
# export OMP_PLACES=cores
set -x
srun -n ${num_ranks} ./bind.sh python sandbox_gaussian_spatioTemporal_medSize.py >output_gaussian_spatioTemporal_medSize_pinned_numaCtl_gpu_numRanks${num_ranks}.txt
#srun -n ${num_ranks} numactl -l --all --physcpubind=0-${OPENBLAS_NUM_THREADS} ./nsys.sh python sandbox_gaussian_spatioTemporal_medSize.py >output_gaussian_spatioTemporal_medSize_pinned_numaCtl_omp${OMP_NUM_THREADS}_numRanks${num_ranks}.txt
set +x
