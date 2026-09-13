#!/bin/bash
#SBATCH --job-name=hpointstd
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=01:00:00
#SBATCH --output=hpointstd.%j.%N.out
#SBATCH --error=hpointstd.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SOURCE_COMMIT="${1:-}"
if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

DATA_ROOT="${2:-$SCRATCH_DSS/hrhsa-surface-scaling}"
MODE="${3:-}"
GEOMETRY="${4:-8x14}"
FIXED_POINTS="${5:-}"
RASTER_GIB="${6:-224}"
POINTS_PER_CORE="${7:-2500000}"
GIB_PER_CORE="${8:-2}"
NODES="$SLURM_JOB_NUM_NODES"

if (( NODES < 2 || NODES > 4 )); then
    echo "cm4_std point scaling requires 2-4 nodes; got $NODES." >&2
    exit 2
fi
if [[ "$MODE" != "strong" && "$MODE" != "weak" ]]; then
    echo "Argument 3 must be strong or weak for cm4_std point scaling." >&2
    exit 2
fi
if [[ ! -d "$DATA_ROOT/environment.zarr" || ! -f "$DATA_ROOT/surface_family_manifest.json" || ! -f "$DATA_ROOT/point_family_manifest.json" ]]; then
    echo "Missing raster and/or point benchmark family under: $DATA_ROOT" >&2
    exit 2
fi

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

OUT="$HRHSA_RESULT_ROOT/cm4-point-${MODE}-n${NODES}-${SLURM_JOB_ID}.jsonl"
COMMON=(
    --root "$DATA_ROOT"
    --output "$OUT"
    --mode "$MODE"
    --backend slurm
    --geometry "$GEOMETRY"
    --node-counts "$NODES"
    --warmup-points 1000000
    --warmup-repeats 1
    --repeats 3
    --spatial-chunk 1024
    --chunk-mb 256
    --worker-startup-timeout 600
)

if [[ "$MODE" == "strong" ]]; then
    if [[ -z "$FIXED_POINTS" ]]; then
        echo "strong mode requires fixed point count as argument 5." >&2
        exit 2
    fi
    EXTRA=(--fixed-points "$FIXED_POINTS" --raster-gib "$RASTER_GIB")
else
    EXTRA=(--points-per-core "$POINTS_PER_CORE" --gib-per-core "$GIB_PER_CORE")
fi

echo "=== CoolMUC-4 ${NODES}-node point ${MODE} scaling ==="
echo "job=$SLURM_JOB_ID nodelist=$SLURM_JOB_NODELIST"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT geometry=$GEOMETRY"

python benchmarking/scripts/01_python/run_point_scaling.py \
    "${COMMON[@]}" \
    "${EXTRA[@]}" \
    "${INTERFACE_ARGS[@]}"

echo "Point scaling results: $OUT"
