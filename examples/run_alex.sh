#!/bin/bash

#SBATCH --job-name=profile_pyinla
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
# ##SBATCH --qos=a100multi
# ##SBATCH --exclusive
#SBATCH --error=%x.err          #The .error file name
#SBATCH --output=%x.out         #The .output file name

num_ranks=2

backend=cupy
export ARRAY_MODULE=${backend}

export MPI_CUDA_AWARE=0
export USE_NCCL=0

TIMESTAMP=$(date +"%H-%M-%S")


# srun -n ${num_ranks} --oversubscribe python coreg_small/run_spatial.py >output_coreg_small_spatial_${backend}_mpi${num_ranks}.txt
srun -n ${num_ranks} --oversubscribe python coreg_small/run.py >output_coreg_small_mpi${num_ranks}_2pS.txt

#srun -n ${num_ranks} --oversubscribe python gr/run.py >output_gr_mpi${num_ranks}.txt

#srun nsys profile --force-overwrite=true -o profile_output_coreg_${TIMESTAMP} python coreg_small/run.py  >output_coreg_weakScaling1_${backend}_nsys_mpi${num_ranks}.txt
# srun nsys profile --force-overwrite=true -o profile_gst_medium_${TIMESTAMP} python gst_medium/run.py  >output_gst_medium_nsys_mpi${num_ranks}.txt
#srun nsys profile --force-overwrite=true -o profile_gst_large_${TIMESTAMP} python gst_large/run.py  >output_gst_large_nsys_mpi${num_ranks}.txt