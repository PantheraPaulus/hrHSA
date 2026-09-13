#!/bin/bash
#SBATCH --job-name=hpgrid16
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:45:00
#SBATCH --output=hpgrid16.%j.%N.out
#SBATCH --error=hpgrid16.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_COMMIT="${1:-}"
DATA_ROOT="${2:-$SCRATCH_DSS/hrhsa-point-capacity-final}"
POINT_COUNT="${3:-1000000000}"
RASTER_GIB="${4:-192}"
REPEATS="${5:-1}"

if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

RESULT_DIR="$HRHSA_RESULT_ROOT/point-spatial-grid16-${SLURM_JOB_ID}"
BASELINE_OUT="$RESULT_DIR/monolithic.jsonl"
mkdir -p "$RESULT_DIR"
rm -f "$BASELINE_OUT"

echo "=== CoolMUC-4 balanced spatial grid A/B ==="
echo "job=$SLURM_JOB_ID"
echo "node=$SLURM_JOB_NODELIST"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "point_count=$POINT_COUNT"
echo "raster_gib=$RASTER_GIB"
echo "A=monolithic 8x14 graph=256"
echo "B=8 grid shards 1x14 each, grid=1x8, graph=32 each"
echo "C=16 grid shards 1x7 each, grid=2x8, graph=16 each"
echo "repeats=$REPEATS"
echo "result_dir=$RESULT_DIR"

echo
echo "============================================================"
echo "A: monolithic baseline"
echo "============================================================"
python benchmarking/scripts/01_python/profile_point_partitioned_production_slurm.py \
    --root "$DATA_ROOT" \
    --output "$BASELINE_OUT" \
    --point-count "$POINT_COUNT" \
    --raster-gib "$RASTER_GIB" \
    --geometry 8x14 \
    --graph-partitions 256 \
    --nodes 1 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --validation-points 100000 \
    --worker-startup-timeout 600 \
    --repeats "$REPEATS"

run_grid () {
    local LABEL="$1"
    local ROWS="$2"
    local COLS="$3"
    local THREADS="$4"
    local GRAPH="$5"
    local MEMORY="$6"
    local COUNT=$((ROWS * COLS))
    local SHARD_DIR="$RESULT_DIR/shards-${LABEL}"
    local SUMMARY="$RESULT_DIR/summary-${LABEL}.jsonl"
    rm -rf "$SHARD_DIR"
    rm -f "$SUMMARY"
    mkdir -p "$SHARD_DIR"

    echo
    echo "============================================================"
    echo "$COUNT balanced grid shards: grid=${ROWS}x${COLS}, each=1x${THREADS}, graph=$GRAPH, memory=${MEMORY}GiB"
    echo "============================================================"

    python benchmarking/scripts/01_python/launch_point_spatial_grid_shards.py \
        --root "$DATA_ROOT" \
        --output-dir "$SHARD_DIR" \
        --point-count "$POINT_COUNT" \
        --raster-gib "$RASTER_GIB" \
        --shard-rows "$ROWS" \
        --shard-cols "$COLS" \
        --workers-per-shard 1 \
        --threads-per-worker "$THREADS" \
        --graph-partitions "$GRAPH" \
        --spatial-chunk 1024 \
        --managed-memory-gib "$MEMORY" \
        --validation-points 100000 \
        --worker-startup-timeout 600 \
        --repeats "$REPEATS"

    python benchmarking/scripts/01_python/summarize_point_spatial_shards.py \
        --shard-dir "$SHARD_DIR" \
        --baseline "$BASELINE_OUT" \
        --output "$SUMMARY"

    cat "$SUMMARY"
}

# Same 8-shard spatial decomposition as the previous winner, now through the
# rectangular-grid implementation so the 16-shard comparison differs only by
# doubling the number of domains and halving cores/graph budget per domain.
run_grid 8 1 8 14 32 24
run_grid 16 2 8 7 16 12

echo
echo "============================================================"
echo "Comparison"
echo "============================================================"
python - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = json.loads(next(line for line in (root / "monolithic.jsonl").read_text().splitlines() if line.strip()))
print(
    f"domains=1 wall={base['pipeline_seconds']:.3f}s "
    f"throughput={base['throughput_points_s']/1e6:.3f}M/s "
    f"busy_cores={base.get('worker_busy_cores_pipeline')}"
)
for label in (8, 16):
    path = root / f"summary-{label}.jsonl"
    record = json.loads(next(line for line in path.read_text().splitlines() if line.strip()))
    print(
        f"domains={label} wall={record['pipeline_seconds']:.3f}s "
        f"throughput={record['throughput_points_s']/1e6:.3f}M/s "
        f"busy_cores={record.get('worker_busy_cores_pipeline')} "
        f"task_parallelism={record.get('task_compute_parallelism_approx')} "
        f"transfer_s={record.get('task_transfer_seconds')} "
        f"speedup={record.get('speedup_vs_monolithic')}"
    )
PY

echo
echo "Completed: $RESULT_DIR"
