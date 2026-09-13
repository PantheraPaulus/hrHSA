#!/bin/bash
#SBATCH --job-name=hprepscale
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=hprepscale.%j.%N.out
#SBATCH --error=hprepscale.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

SOURCE_COMMIT="${1:-}"
DATA_ROOT="${2:-$SCRATCH_DSS/hrhsa-surface-scaling}"
RASTER_GIB="${3:-24,48,96,192,224,448,896}"

if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

echo "=== Prepare nested surface scaling family ==="
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "raster_gib=$RASTER_GIB"

python benchmarking/scripts/01_python/prepare_surface_family.py \
    --root "$DATA_ROOT" \
    --raster-gib "$RASTER_GIB" \
    --bands 6 \
    --storage-chunk 1024 \
    --workers "$SLURM_CPUS_PER_TASK"
