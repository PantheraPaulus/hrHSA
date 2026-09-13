#!/bin/bash
#SBATCH --job-name=hpbal28
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:45:00
#SBATCH --output=hpbal28.%j.%N.out
#SBATCH --error=hpbal28.%j.%N.err
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
SHARDS=28
THREADS=4
GRAPH=9
MEMORY_PER_SHARD=$(python - <<'PY'
print(192.0 / 28.0)
PY
)

if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

if [[ ! -d "$DATA_ROOT/environment.zarr" \
      || ! -f "$DATA_ROOT/surface_family_manifest.json" \
      || ! -f "$DATA_ROOT/point_family_manifest.json" ]]; then
    echo "Missing point/surface benchmark family under: $DATA_ROOT" >&2
    exit 2
fi

RESULT_DIR="$HRHSA_RESULT_ROOT/point-spatial-balanced28-${SLURM_JOB_ID}"
SHARD_DIR="$RESULT_DIR/shards-28"
SUMMARY="$RESULT_DIR/summary-28.jsonl"
rm -rf "$RESULT_DIR"
mkdir -p "$SHARD_DIR"

echo "=== CoolMUC-4 workload-balanced 28-domain point sampling ==="
echo "job=$SLURM_JOB_ID"
echo "node=$SLURM_JOB_NODELIST"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "point_count=$POINT_COUNT"
echo "raster_gib=$RASTER_GIB"
echo "planner=recursive weighted spatial bisection"
echo "domains=$SHARDS"
echo "each=1 worker x $THREADS threads"
echo "graph/domain=$GRAPH aggregate_graph_budget=$((SHARDS * GRAPH))"
echo "managed_memory/domain=${MEMORY_PER_SHARD}GiB aggregate=192GiB"
echo "repeats=$REPEATS"
echo "result_dir=$RESULT_DIR"
echo

python benchmarking/scripts/01_python/launch_point_spatial_balanced_shards.py \
    --root "$DATA_ROOT" \
    --output-dir "$SHARD_DIR" \
    --point-count "$POINT_COUNT" \
    --raster-gib "$RASTER_GIB" \
    --shard-count "$SHARDS" \
    --workers-per-shard 1 \
    --threads-per-worker "$THREADS" \
    --graph-partitions "$GRAPH" \
    --spatial-chunk 1024 \
    --managed-memory-gib "$MEMORY_PER_SHARD" \
    --validation-points 100000 \
    --worker-startup-timeout 600 \
    --repeats "$REPEATS"

python benchmarking/scripts/01_python/summarize_point_spatial_shards.py \
    --shard-dir "$SHARD_DIR" \
    --output "$SUMMARY"

echo
echo "============================================================"
echo "Balanced 28-domain summary"
echo "============================================================"
cat "$SUMMARY"
python - "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path
record = json.loads(next(line for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()))
print(
    f"domains={record['shard_count']} "
    f"wall={record['pipeline_seconds']:.3f}s "
    f"throughput={record['throughput_points_s']/1e6:.3f}M/s "
    f"busy_cores={record.get('worker_busy_cores_pipeline')} "
    f"utilization={100*record['worker_execution_thread_utilization_fraction']:.1f}% "
    f"task_parallelism={record.get('task_compute_parallelism_approx')} "
    f"transfer_s={record.get('task_transfer_seconds')}"
)
PY

echo
echo "Completed: $RESULT_DIR"
