#!/bin/bash
#SBATCH --job-name=himech
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:25:00
#SBATCH --output=himech.%j.%N.out
#SBATCH --error=himech.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# Preserve the submitting checkout revision even when compute nodes do not have
# git available. Pass it as the first sbatch script argument.
SOURCE_COMMIT="${1:-}"
if [[ -z "$SOURCE_COMMIT" ]] && command -v git >/dev/null 2>&1; then
    SOURCE_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
fi
if [[ -n "$SOURCE_COMMIT" ]]; then
    export HRHSA_GIT_COMMIT="$SOURCE_COMMIT"
fi

if [[ ! -d "$HRHSA_LARGE_DATA_ROOT/environment.zarr" ]]; then
    echo "Missing large benchmark raster: $HRHSA_LARGE_DATA_ROOT/environment.zarr" >&2
    exit 2
fi

OUT="$HRHSA_RESULT_ROOT/inter-mechanisms-${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES nodelist=$SLURM_JOB_NODELIST" > "$OUT/allocation.txt"
echo "partition=$SLURM_JOB_PARTITION" >> "$OUT/allocation.txt"
echo "source_commit=${HRHSA_GIT_COMMIT:-unavailable}" >> "$OUT/allocation.txt"

# Static node provenance. Per-worker physical topology is captured separately by
# the strict manifests of every production Dask geometry.
{
  lscpu || true
  echo "--- caches ---"
  lscpu -C || true
  echo "--- topology ---"
  lscpu -e=CPU,CORE,SOCKET,NODE,CACHE,ONLINE,MAXMHZ,MINMHZ || true
  echo "--- numa ---"
  numactl --hardware || true
  echo "--- memory ---"
  free -h || true
  echo "--- scratch ---"
  df -hT "$SCRATCH_DSS" "${TMPDIR:-/tmp}" || true
  echo "--- perf ---"
  perf --version 2>/dev/null || true
  cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || true
} > "$OUT/hardware.txt" 2>&1

# Direct, storage-free production-kernel scaling on resident blocks. This asks
# whether the scientific NumPy kernel itself saturates before Dask enters the
# picture.
DIRECT_ROOT="$OUT/direct-root"
mkdir -p "$DIRECT_ROOT"
ln -sfn "$HRHSA_LARGE_DATA_ROOT" "$DIRECT_ROOT/data"

srun --exact --nodes=1 --ntasks=1 --cpus-per-task=112 --cpu-bind=cores \
  python benchmarking/scripts/01_python/diagnose_numpy_pinned_scaling.py \
    --root "$DIRECT_ROOT" \
    --output "$OUT/direct-kernel-scaling.jsonl" \
    --threads 1,2,4,7,14,28,56,112 \
    --block-size 1024 \
    --resident-blocks 16 \
    --tasks 2048 \
    --repeats 3 \
    --warmup-tasks 128 \
    --compute-dtype float32

# First-touch memory-bandwidth envelope on the same physical-core axis.
srun --exact --nodes=1 --ntasks=1 --cpus-per-task=112 --cpu-bind=cores \
  python benchmarking/scripts/01_python/diagnose_memory_bandwidth.py \
    --output "$OUT/memory-bandwidth.jsonl" \
    --threads 1,2,4,7,14,28,56,112 \
    --total-gib 12 \
    --iterations 16 \
    --repeats 3

# PMU work is deliberately best-effort. Counter restrictions must not waste the
# interactive allocation or suppress the portable diagnostics below.
bash benchmarking/scripts/03_sh/run_coolmuc4_perf_diagnostics.sh \
  "$DIRECT_ROOT" "$OUT/perf" || true

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

# Same production-kernel mechanism runner as cm4_std, restricted to one node.
# This measures the process/thread geometry phenomenon immediately on inter;
# only the true 1->2-node comparison is deferred to cm4_std.
python benchmarking/scripts/01_python/run_coolmuc4_mechanism_diagnostics.py \
  --root "$HRHSA_LARGE_DATA_ROOT" \
  --output-dir "$OUT/dask" \
  --node-counts 1 \
  --geometries 8x14,16x7,56x2,112x1 \
  --warmup-repeats 1 \
  --repeats 3 \
  --spatial-chunk 1024 \
  --chunk-mb 256 \
  --worker-startup-timeout 300 \
  "${INTERFACE_ARGS[@]}"

find "$OUT" -maxdepth 2 -type f -printf '%P\n' | sort > "$OUT/FILES.txt"
echo "Interactive mechanism diagnostics complete: $OUT"
