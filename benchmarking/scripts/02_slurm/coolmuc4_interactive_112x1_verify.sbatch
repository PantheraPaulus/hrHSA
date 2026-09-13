#!/bin/bash
#SBATCH --job-name=h112verify
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:10:00
#SBATCH --output=h112verify.%j.%N.out
#SBATCH --error=h112verify.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Pass `$(git rev-parse HEAD)` as the first sbatch script argument. This preserves
# source provenance even if git itself is unavailable on the compute node.
if [[ -n "${1:-}" ]]; then
    export HRHSA_GIT_COMMIT="$1"
fi

source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

if [[ ! -d "$HRHSA_LARGE_DATA_ROOT/environment.zarr" ]]; then
    echo "Missing large benchmark raster: $HRHSA_LARGE_DATA_ROOT/environment.zarr" >&2
    exit 2
fi

OUTPUT="$HRHSA_RESULT_ROOT/interactive-surface-112x1-${SLURM_JOB_ID}.jsonl"

echo "=== CoolMUC-4 112x1 topology regression verification ==="
echo "job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST"
echo "SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-unset}"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "output=$OUTPUT"

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

# One timed observation is sufficient here: the primary purpose is to prove that
# 112 workers, each bound to one physical core with both SMT siblings visible,
# pass strict physical-space validation. The later cm4_std mechanism campaign
# remains the formal performance experiment.
python benchmarking/scripts/01_python/run_planned_execution.py \
    --root "$HRHSA_LARGE_DATA_ROOT" \
    --output "$OUTPUT" \
    --benchmarks surface \
    --geometry 112x1 \
    --campaign coolmuc4-interactive-112x1-topology-regression \
    --warmup-repeats 0 \
    --repeats 1 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --worker-startup-timeout 300 \
    --strict-affinity \
    "${INTERFACE_ARGS[@]}"

echo "Completed: $OUTPUT"
echo "Manifest : ${OUTPUT}.manifest.json"
