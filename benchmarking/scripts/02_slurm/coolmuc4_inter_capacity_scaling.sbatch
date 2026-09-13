#!/bin/bash
#SBATCH --job-name=hcapinter
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:30:00
#SBATCH --output=hcapinter.%j.%N.out
#SBATCH --error=hcapinter.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

SOURCE_COMMIT="${1:-}"
DATA_ROOT="${2:-$SCRATCH_DSS/hrhsa-surface-scaling}"
GEOMETRY="${3:-8x14}"
RASTER_GIB="${4:-24,48,96,192}"

if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

if [[ ! -d "$DATA_ROOT/environment.zarr" || ! -f "$DATA_ROOT/surface_family_manifest.json" ]]; then
    echo "Missing nested surface benchmark family under: $DATA_ROOT" >&2
    echo "Prepare it first with coolmuc4_prepare_surface_scaling.sbatch." >&2
    exit 2
fi

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

OUT="$HRHSA_RESULT_ROOT/inter-surface-capacity-${SLURM_JOB_ID}.jsonl"

echo "=== cm4_inter instrumented surface capacity scaling ==="
echo "job=$SLURM_JOB_ID"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "geometry=$GEOMETRY"
echo "raster_gib=$RASTER_GIB"
echo "output=$OUT"

python benchmarking/scripts/01_python/run_surface_scaling_instrumented.py \
    --root "$DATA_ROOT" \
    --output "$OUT" \
    --mode capacity \
    --backend slurm \
    --geometry "$GEOMETRY" \
    --raster-gib "$RASTER_GIB" \
    --nodes 1 \
    --warmup-repeats 1 \
    --repeats 3 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --worker-startup-timeout 600 \
    "${INTERFACE_ARGS[@]}"

echo "Completed: $OUT"
echo "Diagnostics: ${OUT}.diagnostics.jsonl"
echo "Keep cm4_inter timings labelled as diagnostics; production confirmation belongs on cm4_std."
