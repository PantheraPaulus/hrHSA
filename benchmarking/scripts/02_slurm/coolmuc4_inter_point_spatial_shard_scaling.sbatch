#!/bin/bash
#SBATCH --job-name=hpshscale
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:45:00
#SBATCH --output=hpshscale.%j.%N.out
#SBATCH --error=hpshscale.%j.%N.err
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
SHARD_COUNTS="${5:-2,4,8}"
TOTAL_GRAPH="${6:-256}"
REPEATS="${7:-1}"
TOTAL_WORKERS=8
THREADS_PER_WORKER=14
TOTAL_MANAGED_MEMORY_GIB=192

if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

if [[ ! -d "$DATA_ROOT/environment.zarr" \
      || ! -f "$DATA_ROOT/surface_family_manifest.json" \
      || ! -f "$DATA_ROOT/point_family_manifest.json" ]]; then
    echo "Missing point/surface benchmark family under: $DATA_ROOT" >&2
    exit 2
fi

RESULT_DIR="$HRHSA_RESULT_ROOT/point-spatial-shard-scaling-${SLURM_JOB_ID}"
BASELINE_OUT="$RESULT_DIR/monolithic.jsonl"
mkdir -p "$RESULT_DIR"
rm -f "$BASELINE_OUT"

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

echo "=== CoolMUC-4 fixed-resource point spatial-shard scaling ==="
echo "job=$SLURM_JOB_ID"
echo "node=$SLURM_JOB_NODELIST"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "point_count=$POINT_COUNT"
echo "raster_gib=$RASTER_GIB"
echo "baseline=8x14 graph=$TOTAL_GRAPH"
echo "shard_counts=$SHARD_COUNTS"
echo "fixed total workers=$TOTAL_WORKERS threads/worker=$THREADS_PER_WORKER"
echo "fixed total graph budget=$TOTAL_GRAPH"
echo "fixed total managed-memory budget=${TOTAL_MANAGED_MEMORY_GIB}GiB"
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
    --graph-partitions "$TOTAL_GRAPH" \
    --nodes 1 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --validation-points 100000 \
    --worker-startup-timeout 600 \
    --repeats "$REPEATS" \
    "${INTERFACE_ARGS[@]}"

IFS=',' read -r -a SHARD_ARRAY <<< "$SHARD_COUNTS"
for SHARDS in "${SHARD_ARRAY[@]}"; do
    if (( SHARDS <= 1 )); then
        echo "Skipping shard_count=$SHARDS; monolithic baseline already represents one domain."
        continue
    fi
    if (( TOTAL_WORKERS % SHARDS != 0 )); then
        echo "Shard count $SHARDS does not divide total workers $TOTAL_WORKERS." >&2
        exit 2
    fi
    if (( TOTAL_GRAPH % SHARDS != 0 )); then
        echo "Shard count $SHARDS does not divide total graph budget $TOTAL_GRAPH." >&2
        exit 2
    fi
    if (( TOTAL_MANAGED_MEMORY_GIB % SHARDS != 0 )); then
        echo "Shard count $SHARDS does not divide managed-memory budget $TOTAL_MANAGED_MEMORY_GIB." >&2
        exit 2
    fi

    WORKERS_PER_SHARD=$((TOTAL_WORKERS / SHARDS))
    GRAPH_PER_SHARD=$((TOTAL_GRAPH / SHARDS))
    MEMORY_PER_SHARD=$((TOTAL_MANAGED_MEMORY_GIB / SHARDS))
    SHARD_DIR="$RESULT_DIR/shards-${SHARDS}"
    SUMMARY="$RESULT_DIR/summary-${SHARDS}.jsonl"
    rm -rf "$SHARD_DIR"
    rm -f "$SUMMARY"
    mkdir -p "$SHARD_DIR"

    echo
    echo "============================================================"
    echo "B: $SHARDS concurrent spatial shards"
    echo "each=${WORKERS_PER_SHARD}x${THREADS_PER_WORKER} graph=$GRAPH_PER_SHARD managed_memory=${MEMORY_PER_SHARD}GiB"
    echo "============================================================"

    python benchmarking/scripts/01_python/launch_point_spatial_shards.py \
        --root "$DATA_ROOT" \
        --output-dir "$SHARD_DIR" \
        --point-count "$POINT_COUNT" \
        --raster-gib "$RASTER_GIB" \
        --shard-count "$SHARDS" \
        --workers-per-shard "$WORKERS_PER_SHARD" \
        --threads-per-worker "$THREADS_PER_WORKER" \
        --graph-partitions "$GRAPH_PER_SHARD" \
        --spatial-chunk 1024 \
        --managed-memory-gib "$MEMORY_PER_SHARD" \
        --validation-points 100000 \
        --worker-startup-timeout 600 \
        --repeats "$REPEATS"

    python benchmarking/scripts/01_python/summarize_point_spatial_shards.py \
        --shard-dir "$SHARD_DIR" \
        --baseline "$BASELINE_OUT" \
        --output "$SUMMARY"

    echo "summary ($SHARDS shards):"
    cat "$SUMMARY"
done

echo
echo "============================================================"
echo "Shard-count comparison"
echo "============================================================"
python - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = [json.loads(line) for line in (root / "monolithic.jsonl").read_text().splitlines() if line.strip()]
for record in base:
    print(
        f"shards=1 repeat={record.get('repeat')} "
        f"wall={record['pipeline_seconds']:.3f}s "
        f"throughput={record['throughput_points_s']/1e6:.3f}M/s "
        f"busy_cores={record.get('worker_busy_cores_pipeline')}"
    )
for path in sorted(root.glob("summary-*.jsonl"), key=lambda p: int(p.stem.split('-')[-1])):
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        print(
            f"shards={record['shard_count']} repeat={record['repeat']} "
            f"wall={record['pipeline_seconds']:.3f}s "
            f"throughput={record['throughput_points_s']/1e6:.3f}M/s "
            f"busy_cores={record.get('worker_busy_cores_pipeline')} "
            f"speedup={record.get('speedup_vs_monolithic')}"
        )
PY

echo
echo "Completed: $RESULT_DIR"
