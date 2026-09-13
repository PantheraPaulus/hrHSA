#!/bin/bash
#SBATCH --job-name=hintergeom
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:30:00
#SBATCH --output=hintergeom.%j.%N.out
#SBATCH --error=hintergeom.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

if [[ ! -d "$HRHSA_LARGE_DATA_ROOT/environment.zarr" ]]; then
    echo "Missing large benchmark raster: $HRHSA_LARGE_DATA_ROOT/environment.zarr" >&2
    echo "Prepare it first with benchmarking/scripts/02_slurm/coolmuc4_prepare_large.sbatch." >&2
    exit 2
fi
if [[ ! -f "$HRHSA_LARGE_DATA_ROOT/manifest.json" ]]; then
    echo "Missing large benchmark manifest: $HRHSA_LARGE_DATA_ROOT/manifest.json" >&2
    exit 2
fi

# Validate the fixed workload before spending an interactive full-node slot.
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["HRHSA_LARGE_DATA_ROOT"]) / "manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
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
print("Large benchmark dataset OK:", path)
print(json.dumps(manifest, indent=2))
PY

# One allocation should compare the leading candidates under the same node/time
# conditions. Extra geometries may be supplied as sbatch arguments, e.g.
#   sbatch ... 8x14 16x7 56x2 112x1
if (( $# == 0 )); then
    GEOMETRIES=("8x14" "16x7")
else
    GEOMETRIES=("$@")
fi

for GEOMETRY in "${GEOMETRIES[@]}"; do
    case "$GEOMETRY" in
        8x14|16x7|14x8|28x4|56x2|112x1) ;;
        *)
            echo "Unsupported full-node geometry: $GEOMETRY" >&2
            echo "Allowed: 8x14 16x7 14x8 28x4 56x2 112x1" >&2
            exit 2
            ;;
    esac

done

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

echo "=== CoolMUC-4 interactive surface campaign ==="
echo "job=$SLURM_JOB_ID node(s)=$SLURM_JOB_NUM_NODES"
echo "SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-unset}"
echo "SLURM_TASKS_PER_NODE=${SLURM_TASKS_PER_NODE:-unset}"
echo "geometries=${GEOMETRIES[*]}"
echo "dataset=$HRHSA_LARGE_DATA_ROOT"
echo "results=$HRHSA_RESULT_ROOT"

for GEOMETRY in "${GEOMETRIES[@]}"; do
    OUTPUT="$HRHSA_RESULT_ROOT/interactive-surface-${GEOMETRY}-${SLURM_JOB_ID}.jsonl"
    echo
    echo "=== geometry $GEOMETRY ==="
    echo "output=$OUTPUT"

    # --cpu-bind=cores may expose both SMT sibling CPU ids in each worker cpuset.
    # Strict validation therefore works in (socket, physical-core) space; the
    # logical cpuset remains recorded as provenance.
    python benchmarking/scripts/01_python/run_planned_execution.py \
        --root "$HRHSA_LARGE_DATA_ROOT" \
        --output "$OUTPUT" \
        --benchmarks surface \
        --geometry "$GEOMETRY" \
        --campaign coolmuc4-interactive-geometry-2026-08-31 \
        --warmup-repeats 1 \
        --repeats 5 \
        --spatial-chunk 1024 \
        --chunk-mb 256 \
        --worker-startup-timeout 600 \
        --strict-affinity \
        "${INTERFACE_ARGS[@]}"

    echo "Completed: $OUTPUT"
    echo "Manifest : ${OUTPUT}.manifest.json"
done

echo
echo "Interactive campaign complete."
echo "Keep cm4_inter timings labelled as diagnostics; production confirmation belongs on cm4_tiny/cm4_std."
