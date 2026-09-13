#!/bin/bash
#SBATCH --job-name=hptunpts
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --output=hptunpts.%j.%N.out
#SBATCH --error=hptunpts.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SOURCE_COMMIT="${1:-}"
DATA_ROOT="${2:-$SCRATCH_DSS/hrhsa-point-capacity-final}"
POINT_COUNT="${3:-1000000000}"
RASTER_GIB="${4:-192}"
GEOMETRIES="${5:-8x14,16x7,28x4,56x2,112x1}"
GRAPH_WINDOWS="${6:-100,256,512}"
REPEATS="${7:-1}"
SKIP_EXISTING_BASELINE="${8:-1}"

if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

if [[ ! -d "$DATA_ROOT/environment.zarr" \
      || ! -f "$DATA_ROOT/surface_family_manifest.json" \
      || ! -f "$DATA_ROOT/point_family_manifest.json" ]]; then
    echo "Missing point/surface benchmark family under: $DATA_ROOT" >&2
    exit 2
fi

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

OUT="$HRHSA_RESULT_ROOT/inter-point-partitioned-tuning-${SLURM_JOB_ID}.jsonl"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

IFS=',' read -r -a GEOMETRY_ARRAY <<< "$GEOMETRIES"
IFS=',' read -r -a GRAPH_ARRAY <<< "$GRAPH_WINDOWS"

echo "=== cm4_inter point-sampling tuning sweep ==="
echo "job=$SLURM_JOB_ID"
echo "node=$SLURM_JOB_NODELIST"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "point_count=$POINT_COUNT"
echo "raster_gib=$RASTER_GIB"
echo "geometries=$GEOMETRIES"
echo "graph_windows=$GRAPH_WINDOWS"
echo "repeats=$REPEATS"
echo "skip_existing_baseline=$SKIP_EXISTING_BASELINE"
echo "output=$OUT"

# Screen the interaction between process/thread geometry and Dask-native point
# graph depth. The 8x14 / graph=100 cell is already available from the frozen
# capacity campaign, so skip it by default to conserve cm4_inter walltime.
for GEOMETRY in "${GEOMETRY_ARRAY[@]}"; do
    for GRAPH in "${GRAPH_ARRAY[@]}"; do
        if [[ "$SKIP_EXISTING_BASELINE" == "1" \
              && "$GEOMETRY" == "8x14" \
              && "$GRAPH" == "100" ]]; then
            echo "Skipping existing capacity baseline: geometry=8x14 graph=100"
            continue
        fi

        echo
        echo "============================================================"
        echo "geometry=$GEOMETRY graph_partitions=$GRAPH"
        echo "points=$POINT_COUNT raster=${RASTER_GIB}GiB"
        echo "============================================================"

        python benchmarking/scripts/01_python/profile_point_partitioned_production_slurm.py \
            --root "$DATA_ROOT" \
            --output "$OUT" \
            --point-count "$POINT_COUNT" \
            --raster-gib "$RASTER_GIB" \
            --geometry "$GEOMETRY" \
            --graph-partitions "$GRAPH" \
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
echo "Use one-repeat results only for screening; confirm the best two settings"
echo "with three repeats before creating a point-specific benchmark profile."
