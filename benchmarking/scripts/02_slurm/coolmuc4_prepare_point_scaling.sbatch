#!/bin/bash
#SBATCH --job-name=hpreppoint
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=hpreppoint.%j.%N.out
#SBATCH --error=hpreppoint.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SOURCE_COMMIT="${1:-}"
if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# Keep the default aligned with coolmuc4_inter_point_partitioned_capacity.sbatch.
# Callers can still override it explicitly as positional argument 2.
DATA_ROOT="${2:-$SCRATCH_DSS/hrhsa-point-capacity-final}"
POINTS="${3:-}"
PARTITION_SIZE="${4:-1000000}"
WORKERS="${5:-16}"

if [[ -z "$POINTS" ]]; then
    echo "Pass comma-separated point targets as argument 3, e.g. 1000000,10000000,100000000." >&2
    exit 2
fi

mkdir -p "$DATA_ROOT"

echo "=== Prepare partitioned point family ==="
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "points=$POINTS"
echo "partition_size=$PARTITION_SIZE"
echo "workers=$WORKERS"

python benchmarking/scripts/01_python/prepare_point_family.py \
    --root "$DATA_ROOT" \
    --points "$POINTS" \
    --partition-size "$PARTITION_SIZE" \
    --workers "$WORKERS"

echo "Point manifest: $DATA_ROOT/point_family_manifest.json"
