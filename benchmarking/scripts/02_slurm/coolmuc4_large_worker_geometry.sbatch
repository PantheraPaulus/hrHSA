#!/bin/bash
#SBATCH --job-name=hlggeom
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:30:00
#SBATCH --output=hlggeom.%j.%N.out
#SBATCH --error=hlggeom.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# This experiment reuses the 65,536 x 65,536 x 6 large raster prepared for the
# multi-node campaign. Fail before starting Dask if the reusable dataset is not
# present or does not match the intended 96-GiB / 4096-chunk workload.
if [[ ! -d "$HRHSA_LARGE_DATA_ROOT/environment.zarr" ]]; then
    echo "Missing large benchmark raster: $HRHSA_LARGE_DATA_ROOT/environment.zarr" >&2
    echo "Prepare it first with benchmarking/scripts/02_slurm/coolmuc4_prepare_large.sbatch." >&2
    exit 2
fi
if [[ ! -f "$HRHSA_LARGE_DATA_ROOT/manifest.json" ]]; then
    echo "Missing large benchmark manifest: $HRHSA_LARGE_DATA_ROOT/manifest.json" >&2
    exit 2
fi

python - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["HRHSA_LARGE_DATA_ROOT"]) / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "size": 65536,
    "bands": 6,
    "storage_chunk": 1024,
    "spatial_chunks": 4096,
}
for key, value in expected.items():
    observed = manifest.get(key)
    if observed != value:
        raise SystemExit(
            f"Large benchmark manifest mismatch for {key}: expected {value}, got {observed}"
        )
print("Large benchmark dataset OK:", manifest_path)
PY

SCHEDULER_FILE="$HRHSA_RESULT_ROOT/large-geometry-scheduler-${SLURM_JOB_ID}.json"
OUTPUT="$HRHSA_RESULT_ROOT/large-worker-geometry-${SLURM_JOB_ID}.jsonl"
rm -f "$SCHEDULER_FILE"

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

SCHED_PID=""
WORKER_STEP_PID=""
cleanup() {
    set +e
    [[ -n "$WORKER_STEP_PID" ]] && kill "$WORKER_STEP_PID" 2>/dev/null
    [[ -n "$SCHED_PID" ]] && kill "$SCHED_PID" 2>/dev/null
    rm -f "$SCHEDULER_FILE"
}
trap cleanup EXIT

dask scheduler \
    --scheduler-file "$SCHEDULER_FILE" \
    --dashboard-address :0 \
    "${INTERFACE_ARGS[@]}" &
SCHED_PID=$!

for _ in $(seq 1 120); do
    [[ -s "$SCHEDULER_FILE" ]] && break
    sleep 1
done
if [[ ! -s "$SCHEDULER_FILE" ]]; then
    echo "Dask scheduler file was not created." >&2
    exit 3
fi

# The medium-raster geometry campaign identified a broad useful region around
# 8-16 worker processes. Retain 112x1 as the pure-process control and progressively
# group the same 112 physical cores into fewer multithreaded worker processes.
# Every geometry uses all 112 allocated physical cores.
GEOMETRIES=(
    "112:1"
    "56:2"
    "28:4"
    "16:7"
    "14:8"
    "8:14"
)

# Keep aggregate Dask-managed memory below the 224-GiB Slurm request and leave
# headroom for Python, the scheduler, page cache and unmanaged arrays.
DASK_MANAGED_TOTAL_GIB="${HRHSA_DASK_MANAGED_TOTAL_GIB:-192}"

for geometry in "${GEOMETRIES[@]}"; do
    IFS=: read -r WORKERS THREADS <<<"$geometry"
    MEM_LIMIT=$(python - <<PY
workers = int("$WORKERS")
total = float("$DASK_MANAGED_TOTAL_GIB")
print(f"{total / workers:.3f}GiB")
PY
)
    export SCHEDULER_FILE THREADS MEM_LIMIT

    echo "=== large geometry ${WORKERS} workers x ${THREADS} threads; memory=${MEM_LIMIT}/worker ==="

    srun \
        --exact \
        --nodes=1 \
        --ntasks="$WORKERS" \
        --cpus-per-task="$THREADS" \
        --distribution=block:block \
        --cpu-bind=cores \
        bash -lc 'args=(); if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then args=(--interface "$HRHSA_DASK_INTERFACE"); fi; exec dask worker --scheduler-file "$SCHEDULER_FILE" --nthreads "$THREADS" --memory-limit "$MEM_LIMIT" --local-directory "$TMPDIR/dask-${SLURM_PROCID}" "${args[@]}"' &
    WORKER_STEP_PID=$!

    python benchmarking/scripts/01_python/run_worker_geometry.py \
        --root "$HRHSA_LARGE_DATA_ROOT" \
        --output "$OUTPUT" \
        --scheduler-file "$SCHEDULER_FILE" \
        --workers "$WORKERS" \
        --threads-per-worker "$THREADS" \
        --benchmarks surface \
        --repeats 3 \
        --spatial-chunk 1024 \
        --chunk-mb 256 \
        --worker-startup-timeout 600

    kill "$WORKER_STEP_PID" 2>/dev/null || true
    wait "$WORKER_STEP_PID" 2>/dev/null || true
    WORKER_STEP_PID=""

    # This polls only the private Dask scheduler, not Slurm.
    python - <<PY
from dask.distributed import Client
import time

client = Client(scheduler_file="$SCHEDULER_FILE")
try:
    for _ in range(120):
        if not client.scheduler_info().get("workers", {}):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("Previous Dask workers did not retire in time")
finally:
    client.close()
PY
done

echo "Large geometry results: $OUTPUT"