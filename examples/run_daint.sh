#!/bin/bash -l
#SBATCH --job-name="examples"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=sm96
#SBATCH --time=00:29:00
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

#source ~/setup_pyinla.sh
#conda activate pyinla

source ~/load_modules.sh
conda activate allin

export ARRAY_MODULE=cupy
export MPI_CUDA_AWARE=0

# srun python /users/vmaillou/repositories/pyINLA/examples/gst_small/run.py
# srun python /users/vmaillou/repositories/pyINLA/examples/gst_medium/run.py
# srun python /users/vmaillou/repositories/pyINLA/examples/gst_large/run.py
srun python /users/vmaillou/repositories/pyINLA/examples/coreg_small/run.py



# num_ranks=18
# srun -n 18 --oversubscribe python ~/repositories/pyINLA/examples/gst_small/run.py

#srun python ~/repositories/pyINLA/examples/gst_medium/run.py
# srun nsys profile --force-overwrite=true -o profile_gst_medium python ~/repositories/pyINLA/examples/gst_medium/run.py

#srun python ~/pyINLA/examples/coreg_small/run.py
# srun python ~/repositories/pyINLA/examples/coreg_small/run.py

# srun python ~/repositories/pyINLA/runs_sc25/scientific_run/run.py

# srun nsys profile --force-overwrite=true -o profile_ss_large python ~/repositories/pyINLA/examples/gst_large/run.py
#srun nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --force-overwrite=true -o profile_strong_scaling python ~/repositories/pyINLA/examples/coreg_small/run.py

