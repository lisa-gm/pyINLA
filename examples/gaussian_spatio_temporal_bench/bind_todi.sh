#!/bin/bash

set -x
set -euo pipefail

cpu_st=$((0+72*SLURM_LOCALID))
cpu_end=$((cpu_st+OPENBLAS_NUM_THREADS))

numactl -l --all --physcpubind=${cpu_st}-${cpu_end} "$@"
