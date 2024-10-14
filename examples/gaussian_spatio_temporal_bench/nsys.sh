#!/bin/bash

set -x
set -euo pipefail

hwloc-bind --get --taskset 

rank="${OMPI_COMM_WORLD_RANK:-$SLURM_PROCID}"
size="${OMPI_COMM_WORLD_SIZE:-$SLURM_NTASKS}"

#echo "$rank of $size"

[[ -z "${NSYS_FILE+x}" ]] && NSYS_FILE=report.qdrep
[[ -z "${NSYS+x}" ]] && NSYS=0

echo "rank ${rank} $(hostname)" 

if [[ "$NSYS" -ne 0 && "$rank" -eq 0 ]]; then
  exec nsys profile --trace=mpi,nvtx,cuda --force-overwrite=true -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 -o "$NSYS_FILE" "$@"
else
  exec "$@"
fi