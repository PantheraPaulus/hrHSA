#!/bin/bash
#SBATCH --job-name=hplan2
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=hplan2.%j.%N.out
#SBATCH --error=hplan2.%j.%N.err
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

NODES="$SLURM_JOB_NUM_NODES"
if (( NODES < 2 || NODES > 4 )); then
    echo "cm4_std currently permits 2-4 nodes per job; got $NODES." >&2
    exit 2
fi

OUTPUT="$HRHSA_RESULT_ROOT/planned-surface-n${NODES}-${SLURM_JOB_ID}.jsonl"
INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

python benchmarking/scripts/01_python/run_planned_execution.py \
    --root "$HRHSA_LARGE_DATA_ROOT" \
    --output "$OUTPUT" \
    --benchmarks surface \
    --geometry auto \
    --warmup-repeats 1 \
    --repeats 5 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --worker-startup-timeout 600 \
    --strict-affinity \
    "${INTERFACE_ARGS[@]}"

echo "Planned multi-node execution results: $OUTPUT"
echo "Run manifest: ${OUTPUT}.manifest.json"
