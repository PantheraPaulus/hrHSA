#!/bin/bash
#SBATCH --job-name=hpreplrg
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=hpreplrg.%j.%N.out
#SBATCH --error=hpreplrg.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# Surface-only multi-node benchmark: the tiny point table is just a schema/input
# placeholder and is never loaded by run_scaling.py. Thirty-two physical cores
# satisfy the current cm4_tiny minimum of 17 cores. The Zarr writer is streamed,
# so a 128-GiB request leaves substantial headroom without reserving near-node RAM.
python benchmarking/scripts/01_python/prepare_data.py \
    --root "$HRHSA_LARGE_DATA_ROOT" \
    --size 65536 \
    --bands 6 \
    --storage-chunk 1024 \
    --points 1000 \
    --workers "$SLURM_CPUS_PER_TASK"
