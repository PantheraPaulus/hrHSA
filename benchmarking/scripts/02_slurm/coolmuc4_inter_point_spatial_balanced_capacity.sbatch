#!/bin/bash
#SBATCH --job-name=hpbalcap
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --output=hpbalcap.%j.%N.out
#SBATCH --error=hpbalcap.%j.%N.err
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
REPEATS="${5:-3}"

# Frozen CoolMUC-4 point-sampling architecture from the 2026-09-04 tuning
# campaign. Balanced 28 domains were faster than balanced 16 and 56 domains at
# 1B points / ~194 GiB while mapping exactly onto the 112 physical cores.
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

RESULT_DIR="$HRHSA_RESULT_ROOT/point-spatial-balanced-capacity-${RASTER_GIB}gib-${SLURM_JOB_ID}"
SHARD_DIR="$RESULT_DIR/shards-28"
SUMMARY="$RESULT_DIR/summary.jsonl"
LAUNCHER_SUMMARY="$SHARD_DIR/launcher-summary.json"
rm -rf "$RESULT_DIR"
mkdir -p "$SHARD_DIR"

echo "=== CoolMUC-4 balanced point raster-capacity campaign ==="
echo "job=$SLURM_JOB_ID"
echo "node=$SLURM_JOB_NODELIST"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "point_count=$POINT_COUNT"
echo "raster_gib=$RASTER_GIB"
echo "planner=recursive weighted spatial bisection"
echo "domains=$SHARDS"
echo "each=1 worker x $THREADS physical cores"
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
echo "Balanced capacity summary: ${RASTER_GIB} GiB target"
echo "============================================================"
cat "$SUMMARY"
python - "$SUMMARY" "$LAUNCHER_SUMMARY" "$SHARD_DIR" <<'PY'
import json
import statistics
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
launcher_path = Path(sys.argv[2])
shard_dir = Path(sys.argv[3])
records = [
    json.loads(line)
    for line in summary_path.read_text().splitlines()
    if line.strip()
]
launcher = json.loads(launcher_path.read_text())
setup_records = [
    json.loads(path.read_text())
    for path in sorted(shard_dir.glob("shard-*.setup.json"))
]

wall = [float(record["pipeline_seconds"]) for record in records]
throughput = [float(record["throughput_points_s"]) for record in records]
busy = [float(record["worker_busy_cores_pipeline"]) for record in records]
raster = [float(record["raster_gib_full"]) for record in records]
first = min(records, key=lambda record: int(record["repeat"]))
setup_to_barrier = float(launcher["setup_to_barrier_seconds"])
launcher_total = float(launcher["launcher_total_seconds"])
xy_cache_total = sum(float(record.get("xy_cache_seconds", 0.0)) for record in setup_records)
xy_cache_cold = sum(
    float(record.get("xy_cache_seconds", 0.0))
    for record in setup_records
    if not bool(record.get("xy_cache_present_before", True))
)
cache_build_shards = sum(
    not bool(record.get("xy_cache_present_before", True))
    for record in setup_records
)
first_sampling = float(first["pipeline_seconds"])
cold_first_result = setup_to_barrier + first_sampling
production_like_first_result = max(0.0, setup_to_barrier - xy_cache_total) + first_sampling

print(
    f"raster={statistics.mean(raster):.3f}GiB "
    f"wall_mean={statistics.mean(wall):.3f}s "
    f"wall_sd={(statistics.stdev(wall) if len(wall) > 1 else 0.0):.3f}s "
    f"throughput_mean={statistics.mean(throughput)/1e6:.3f}M/s "
    f"busy_cores_mean={statistics.mean(busy):.2f}"
)
print(
    f"setup_to_barrier={setup_to_barrier:.3f}s "
    f"xy_cache_total={xy_cache_total:.3f}s "
    f"xy_cache_cold_build={xy_cache_cold:.3f}s "
    f"cache_build_shards={cache_build_shards}"
)
print(
    f"cold_first_result_including_fixture={cold_first_result:.3f}s "
    f"production_like_first_result_excluding_xy_fixture="
    f"{production_like_first_result:.3f}s "
    f"launcher_total_{len(records)}repeats={launcher_total:.3f}s"
)
PY

echo
echo "Completed: $RESULT_DIR"
echo "Kernel sampling timings and end-to-end timings are both reported. The x/y cache"
echo "is a synthetic benchmark-fixture conversion from reusable normalized u/v points;"
echo "do not interpret that conversion as an intrinsic production sampler cost."
echo "Submit one raster size per Slurm job so each capacity point has an independent"
echo "allocation and the 768-GiB out-of-core case cannot crowd out smaller cases."
