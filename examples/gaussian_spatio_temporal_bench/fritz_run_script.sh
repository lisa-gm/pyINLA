#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=output_gaussian_spatioTemporal_bench.out
#SBATCH --error=output_gaussian_spatioTemporal_bench.err
#SBATCH --time=02:59:00

#export CUDA_VISIBLE_DEVICES=


num_ranks=${SLURM_NTASKS}

#set -x

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=$((OMP_NUM_THREADS-1))

#conda init
#conda deactivate 

#source ~/start_uenv.sh
#source ~/load_modules.sh

#. "/users/hck24/miniconda3/etc/profile.d/conda.sh"

#conda init
conda activate pyinla

# omp${OMP_NUM_THREADS}
export NSYS=1
export NSYS_FILE=gaussian_spatioTemporal_bench_pinned_numaCtl_cpu_omp${OMP_NUM_THREADS}_numRanks${num_ranks}

export PATH=/users/hck24/applications/opt/nvidia/nsight-systems-cli/2024.6.1/target-linux-sbsa-armv8:$PATH

# export OMP_PROC_BIND=true
# export OMP_PLACES=cores
set -x
#srun -n ${num_ranks} python run.py >output_gaussian_spatioTemporal_bench_pinned_numaCtl_cpu_omp${OMP_NUM_THREADS}_numRanks${num_ranks}.txt
#srun -n ${num_ranks} ./bind.sh ./nsys.sh python run.py >output_gaussian_spatioTemporal_bench_pinned_numaCtl_gpu_numRanks${num_ranks}.txt
srun -n ${num_ranks} numactl -l --all --physcpubind=0-${OPENBLAS_NUM_THREADS} python run.py >output_gaussian_spatioTemporal_pinned_numaCtl_cpu_omp${OMP_NUM_THREADS}_numRanks${num_ranks}.txt
set +x
