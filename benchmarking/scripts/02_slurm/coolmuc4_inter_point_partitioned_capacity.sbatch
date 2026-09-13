#!/bin/bash
#SBATCH --job-name=hpcappts
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --output=hpcappts.%j.%N.out
#SBATCH --error=hpcappts.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

SOURCE_COMMIT="${1:-}"
DATA_ROOT="${2:-$SCRATCH_DSS/hrhsa-point-capacity-final}"
GEOMETRY="${3:-8x14}"
GRAPH_PARTITIONS="${4:-256}"
POINT_COUNTS="${5:-1000000,10000000,100000000,1000000000}"
# 384 GiB is intentionally above the 224-GiB Slurm allocation. 192 GiB is
# retained as the transition regime immediately below the managed-memory budget.
RASTER_GIB="${6:-24,96,192,384}"
REPEATS="${7:-3}"

if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

if [[ ! -d "$DATA_ROOT/environment.zarr" \
      || ! -f "$DATA_ROOT/surface_family_manifest.json" \
      || ! -f "$DATA_ROOT/point_family_manifest.json" ]]; then
    echo "Missing point/surface benchmark family under: $DATA_ROOT" >&2
    echo "Prepare the nested raster and spatial point families first." >&2
    exit 2
fi

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

OUT="$HRHSA_RESULT_ROOT/inter-point-partitioned-capacity-${SLURM_JOB_ID}.jsonl"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

IFS=',' read -r -a POINT_ARRAY <<< "$POINT_COUNTS"
IFS=',' read -r -a RASTER_ARRAY <<< "$RASTER_GIB"

echo "=== cm4_inter production partitioned point capacity ==="
echo "job=$SLURM_JOB_ID"
echo "node=$SLURM_JOB_NODELIST"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "geometry=$GEOMETRY"
echo "graph_partitions=$GRAPH_PARTITIONS"
echo "point_counts=$POINT_COUNTS"
echo "raster_gib=$RASTER_GIB"
echo "repeats=$REPEATS"
echo "output=$OUT"

# Keep raster size outermost. This makes each family of point-count observations
# share the same raster window/page-cache regime as closely as practical. For the
# cm4_inter 2-hour limit, submitting one raster size per Slurm job is recommended.
for RASTER in "${RASTER_ARRAY[@]}"; do
    for POINTS in "${POINT_ARRAY[@]}"; do
        echo
        echo "============================================================"
        echo "points=$POINTS raster=${RASTER}GiB"
        echo "============================================================"

        python benchmarking/scripts/01_python/profile_point_partitioned_production_slurm.py \
            --root "$DATA_ROOT" \
            --output "$OUT" \
            --point-count "$POINTS" \
            --raster-gib "$RASTER" \
            --geometry "$GEOMETRY" \
            --graph-partitions "$GRAPH_PARTITIONS" \
            --nodes 1 \
            --spatial-chunk 1024 \
            --chunk-mb 256 \
            --validation-points 100000 \
            --worker-startup-timeout 600 \
            --repeats "$REPEATS" \
            "${INTERFACE_ARGS[@]}"
    done
done

echo
echo "Completed: $OUT"
echo "cm4_inter is a one-node diagnostic/capacity environment; keep these timings"
echo "separate from later cm4_std production/multi-node confirmation."
