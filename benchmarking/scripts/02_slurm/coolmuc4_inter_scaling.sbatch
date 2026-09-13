#!/bin/bash
#SBATCH --job-name=hscaleint
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=01:00:00
#SBATCH --output=hscaleint.%j.%N.out
#SBATCH --error=hscaleint.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# cm4_inter is a single-node partition. This launcher provides the matching
# one-node strong/weak reference points; true 2-4 node scaling belongs in
# coolmuc4_std_surface_scaling.sbatch on cm4_std.
SOURCE_COMMIT="${1:-}"
DATA_ROOT="${2:-$SCRATCH_DSS/hrhsa-surface-scaling}"
GEOMETRY="${3:-8x14}"
STRONG_RASTER_GIB="${4:-224}"
GIB_PER_CORE="${5:-2}"

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

STRONG_OUT="$HRHSA_RESULT_ROOT/inter-surface-strong-${SLURM_JOB_ID}.jsonl"
WEAK_OUT="$HRHSA_RESULT_ROOT/inter-surface-weak-${SLURM_JOB_ID}.jsonl"

echo "=== cm4_inter one-node surface strong/weak references ==="
echo "job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}"
echo "data_root=$DATA_ROOT"
echo "geometry_per_node=$GEOMETRY"
echo "strong_raster_gib=$STRONG_RASTER_GIB"
echo "weak_gib_per_core=$GIB_PER_CORE"

python benchmarking/scripts/01_python/run_surface_scaling_instrumented.py \
    --root "$DATA_ROOT" \
    --output "$STRONG_OUT" \
    --mode strong \
    --backend slurm \
    --geometry "$GEOMETRY" \
    --fixed-raster-gib "$STRONG_RASTER_GIB" \
    --node-counts 1 \
    --warmup-repeats 1 \
    --repeats 3 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --worker-startup-timeout 600 \
    "${INTERFACE_ARGS[@]}"

python benchmarking/scripts/01_python/run_surface_scaling_instrumented.py \
    --root "$DATA_ROOT" \
    --output "$WEAK_OUT" \
    --mode weak \
    --backend slurm \
    --geometry "$GEOMETRY" \
    --node-counts 1 \
    --gib-per-core "$GIB_PER_CORE" \
    --warmup-repeats 1 \
    --repeats 3 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --worker-startup-timeout 600 \
    "${INTERFACE_ARGS[@]}"

echo "strong_results=$STRONG_OUT"
echo "weak_results=$WEAK_OUT"
