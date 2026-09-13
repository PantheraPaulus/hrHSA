#!/bin/bash
#SBATCH --job-name=hpshardab
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:45:00
#SBATCH --output=hpshardab.%j.%N.out
#SBATCH --error=hpshardab.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# Prefer this checkout even if the active environment contains an editable
# install pointing at another sibling hrHSA clone.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_COMMIT="${1:-}"
DATA_ROOT="${2:-$SCRATCH_DSS/hrhsa-point-capacity-final}"
POINT_COUNT="${3:-1000000000}"
RASTER_GIB="${4:-192}"
BASELINE_GRAPH="${5:-256}"
SHARD_GRAPH="${6:-128}"
REPEATS="${7:-1}"

if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

if [[ ! -d "$DATA_ROOT/environment.zarr" \
      || ! -f "$DATA_ROOT/surface_family_manifest.json" \
      || ! -f "$DATA_ROOT/point_family_manifest.json" ]]; then
    echo "Missing point/surface benchmark family under: $DATA_ROOT" >&2
    exit 2
fi

RESULT_DIR="$HRHSA_RESULT_ROOT/point-spatial-shard-ab-${SLURM_JOB_ID}"
BASELINE_OUT="$RESULT_DIR/monolithic.jsonl"
SHARD_DIR="$RESULT_DIR/shards"
SHARD_SUMMARY="$RESULT_DIR/sharded-summary.jsonl"
mkdir -p "$SHARD_DIR"
rm -f "$BASELINE_OUT" "$SHARD_SUMMARY" "$SHARD_DIR"/shard-*.jsonl "$SHARD_DIR"/shard-*.launch.log
rm -rf "$SHARD_DIR/barrier"

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

echo "=== CoolMUC-4 point spatial-shard A/B ==="
echo "job=$SLURM_JOB_ID"
echo "node=$SLURM_JOB_NODELIST"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "point_count=$POINT_COUNT"
echo "raster_gib=$RASTER_GIB"
echo "baseline=8x14 graph=$BASELINE_GRAPH"
echo "sharded=2 concurrent socket shards; each 4x14 graph=$SHARD_GRAPH"
echo "repeats=$REPEATS"
echo "result_dir=$RESULT_DIR"
echo
python - <<'PY'
import hsa
import os
from pathlib import Path

print("hsa import:", hsa.__file__)
cpus = sorted(os.sched_getaffinity(0))
groups = set()
for cpu in cpus:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    groups.add((
        int((base / "physical_package_id").read_text()),
        int((base / "core_id").read_text()),
    ))
print(f"batch affinity: logical_cpus={len(cpus)} physical_cores={len(groups)}")
PY

echo
echo "============================================================"
echo "A: monolithic full-node baseline"
echo "============================================================"
python benchmarking/scripts/01_python/profile_point_partitioned_production_slurm.py \
    --root "$DATA_ROOT" \
    --output "$BASELINE_OUT" \
    --point-count "$POINT_COUNT" \
    --raster-gib "$RASTER_GIB" \
    --geometry "8x14" \
    --graph-partitions "$BASELINE_GRAPH" \
    --nodes 1 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --validation-points 100000 \
    --worker-startup-timeout 600 \
    --repeats "$REPEATS" \
    "${INTERFACE_ARGS[@]}"

echo
echo "============================================================"
echo "B: two concurrent socket-local spatial shards"
echo "============================================================"
# Do not use a nested srun here. On CoolMUC-4, --cpus-per-task in that nested
# step was counted against logical CPU ids while --cpu-bind=cores expanded SMT
# sibling masks, so 56 requested CPUs yielded only 28 physical cores per shard.
# The Python launcher instead starts from the full batch-step cpuset, groups
# logical ids by physical package/core, and explicitly gives each shard one
# 56-physical-core socket.
python benchmarking/scripts/01_python/launch_point_spatial_shards.py \
    --root "$DATA_ROOT" \
    --output-dir "$SHARD_DIR" \
    --point-count "$POINT_COUNT" \
    --raster-gib "$RASTER_GIB" \
    --shard-count 2 \
    --workers-per-shard 4 \
    --threads-per-worker 14 \
    --graph-partitions "$SHARD_GRAPH" \
    --spatial-chunk 1024 \
    --managed-memory-gib 96 \
    --validation-points 100000 \
    --worker-startup-timeout 600 \
    --repeats "$REPEATS"

echo
echo "============================================================"
echo "A/B summary"
echo "============================================================"
python benchmarking/scripts/01_python/summarize_point_spatial_shards.py \
    --shard-dir "$SHARD_DIR" \
    --baseline "$BASELINE_OUT" \
    --output "$SHARD_SUMMARY"

echo
echo "Monolithic records: $BASELINE_OUT"
echo "Shard records      : $SHARD_DIR"
echo "Sharded summary    : $SHARD_SUMMARY"
cat "$SHARD_SUMMARY"
