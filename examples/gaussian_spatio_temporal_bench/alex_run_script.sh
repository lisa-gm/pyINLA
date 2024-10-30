#!/bin/bash
#SBATCH --nodes=2
#SBATCH --qos=a100multi
#SBATCH --ntasks=9
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:5
#SBATCH --constraint=a100_80
#SBATCH --output=output_gaussian_spatioTemporal_bench.out
#SBATCH --error=output_gaussian_spatioTemporal_bench.err
####SBATCH --exclusive
#SBATCH --time=00:29:00


#export CUDA_VISIBLE_DEVICES=


num_ranks=${SLURM_NTASKS}

#set -x

export OMP_NUM_THREADS=16
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
export NSYS_FILE=gaussian_spatioTemporal_bench_pinned_numaCtl_gpu_omp${OMP_NUM_THREADS}_numRanks${num_ranks}

# ...based on the A100 nodes documentation
# GPU : CPU affinity : mask_cpu 
#  0  :    48-63     : 0xffff000000000000
#  1  :    48-63     : 0xffff000000000000
#  2  :    16-31     : 0xffff0000
#  3  :    16-31     : 0xffff0000
#  4  :    112-127   : 0xffff0000000000000000000000000000
#  5  :    112-127   : 0xffff0000000000000000000000000000
#  6  :    80-95     : 0xffff00000000000000000000
#  7  :    80-95     : 0xffff00000000000000000000


# ...modified cpu_mask to use all NUMA domains
# GPU : CPU affinity : mask_cpu 
#  0  :    32-47     : 0xffff00000000
#  1  :    48-63     : 0xffff000000000000
#  2  :    0-15      : 0xffff
#  3  :    16-31     : 0xffff0000
#  4  :    96-111    : 0xffff000000000000000000000000
#  5  :    112-127   : 0xffff0000000000000000000000000000
#  6  :    64-79     : 0xffff0000000000000000
#  7  :    80-95     : 0xffff00000000000000000000

# CPU_BIND="mask_cpu:0xffff00000000,0xffff000000000000"
# CPU_BIND="${CPU_BIND},0xffff,0xffff0000"
# CPU_BIND="${CPU_BIND},0xffff000000000000000000000000,0xffff0000000000000000000000000000"
# CPU_BIND="${CPU_BIND},0xffff0000000000000000,0xffff00000000000000000000"


# export OMP_PROC_BIND=true
# export OMP_PLACES=cores
set -x
srun -n $num_ranks python run.py >output_gaussian_spatioTemporal_bench_pinned_numaCtl_gpu_omp${OMP_NUM_THREADS}_numRanks${num_ranks}.txt
#srun -n ${num_ranks} python run.py >output_gaussian_spatioTemporal_bench_pinned_numaCtl_gpu_omp${OMP_NUM_THREADS}_numRanks${num_ranks}.txt
#srun -n ${num_ranks} ./bind.sh ./nsys.sh python run.py >output_gaussian_spatioTemporal_bench_pinned_numaCtl_gpu_numRanks${num_ranks}.txt
#srun -n ${num_ranks} numactl -l --all --physcpubind=0-${OPENBLAS_NUM_THREADS} ./nsys.sh python sandbox_gaussian_spatioTemporal_medSize.py >output_gaussian_spatioTemporal_medSize_pinned_numaCtl_omp${OMP_NUM_THREADS}_numRanks${num_ranks}.txt
set +x
