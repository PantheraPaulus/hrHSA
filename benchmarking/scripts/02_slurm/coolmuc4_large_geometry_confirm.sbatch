#!/bin/bash
#SBATCH --job-name=hgeomcf
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:30:00
#SBATCH --output=hgeomcf.%j.%N.out
#SBATCH --error=hgeomcf.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

WORKERS="${1:?Usage: sbatch coolmuc4_large_geometry_confirm.sbatch WORKERS THREADS}"
THREADS="${2:?Usage: sbatch coolmuc4_large_geometry_confirm.sbatch WORKERS THREADS}"
GEOMETRY="${WORKERS}x${THREADS}"

case "${WORKERS}:${THREADS}" in
    8:14|16:7|28:4) ;;
    *)
        echo "Confirmatory geometries are restricted to 8x14, 16x7 and 28x4." >&2
        exit 2
        ;;
esac

if (( WORKERS * THREADS != 112 )); then
    echo "Geometry must use exactly 112 physical cores; got ${GEOMETRY}." >&2
    exit 2
fi

if [[ ! -d "$HRHSA_LARGE_DATA_ROOT/environment.zarr" ]]; then
    echo "Missing large benchmark raster: $HRHSA_LARGE_DATA_ROOT/environment.zarr" >&2
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

OUTPUT="$HRHSA_RESULT_ROOT/large-geometry-confirm-${GEOMETRY}-${SLURM_JOB_ID}.jsonl"
INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

echo "=== production geometry confirmation: ${GEOMETRY} ==="
echo "output=$OUTPUT"

# Use exactly the same planner, Slurm worker launcher, physical-core validation,
# manifest/provenance path and production surface kernel as the interactive
# campaign. This avoids a second hand-written Dask launch path drifting away from
# the code we are actually validating.
python benchmarking/scripts/01_python/run_planned_execution.py \
    --root "$HRHSA_LARGE_DATA_ROOT" \
    --output "$OUTPUT" \
    --benchmarks surface \
    --geometry "$GEOMETRY" \
    --campaign coolmuc4-production-geometry-confirm \
    --warmup-repeats 1 \
    --repeats 5 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --worker-startup-timeout 600 \
    --strict-affinity \
    "${INTERFACE_ARGS[@]}"

echo "Confirmatory geometry results: $OUTPUT"
echo "Run manifest: ${OUTPUT}.manifest.json"
