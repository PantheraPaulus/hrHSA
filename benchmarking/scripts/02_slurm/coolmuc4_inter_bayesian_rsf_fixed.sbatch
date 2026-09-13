#!/bin/bash
#SBATCH --job-name=hirsf28
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --hint=nomultithread
#SBATCH --mem=56G
#SBATCH --time=00:06:00
#SBATCH --output=hirsf28.%j.%N.out
#SBATCH --error=hirsf28.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# Short cm4_inter topology diagnostic with fixed statistical model size.
# Unlike LOIO scaling, --models changes only the number of independent fits;
# rows/individuals inside every posterior stay constant. Keep inter timings
# labelled as diagnostics rather than publication timings.
WORKERS="${1:-28}"
CORES_PER_FIT="${2:-1}"
MODELS="${3:-112}"
SEED="${4:-42}"
ROWS="${HRHSA_RSF_FIXED_ROWS:-12000}"
INDIVIDUALS="${HRHSA_RSF_FIXED_INDIVIDUALS:-24}"
PREDICTORS="${HRHSA_RSF_FIXED_PREDICTORS:-3}"
DRAWS="${HRHSA_RSF_FIXED_DRAWS:-250}"
TUNE="${HRHSA_RSF_FIXED_TUNE:-250}"
CHAINS="${HRHSA_RSF_FIXED_CHAINS:-4}"
TARGET_ACCEPT="${HRHSA_RSF_FIXED_TARGET_ACCEPT:-0.9}"

if (( WORKERS <= 0 || CORES_PER_FIT <= 0 || MODELS < WORKERS )); then
    echo "Require positive workers/cores and models >= workers." >&2
    exit 2
fi
if (( WORKERS * CORES_PER_FIT != SLURM_CPUS_PER_TASK )); then
    echo "Allocation mismatch: workers*cores_per_fit=$((WORKERS * CORES_PER_FIT)), allocated=$SLURM_CPUS_PER_TASK" >&2
    exit 2
fi

python - <<'PY'
import importlib.util
import pymc as pm
if importlib.util.find_spec("nutpie") is None:
    raise SystemExit("nutpie is required; install the project with .[bayesian-hpc]")
import nutpie
print("PyMC:", pm.__version__, "nutpie:", nutpie.__version__)
PY

SCHEDULER_FILE="$HRHSA_RESULT_ROOT/inter-rsf-fixed-scheduler-${SLURM_JOB_ID}.json"
OUTPUT="$HRHSA_RESULT_ROOT/inter-bayesian-rsf-fixed-w${WORKERS}x${CORES_PER_FIT}-${SLURM_JOB_ID}.jsonl"
rm -f "$SCHEDULER_FILE" "$OUTPUT"

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

dask scheduler --scheduler-file "$SCHEDULER_FILE" --dashboard-address :0 "${INTERFACE_ARGS[@]}" &
SCHED_PID=$!
for _ in $(seq 1 120); do
    [[ -s "$SCHEDULER_FILE" ]] && break
    sleep 1
done
[[ -s "$SCHEDULER_FILE" ]] || { echo "Dask scheduler file was not created." >&2; exit 3; }

export SCHEDULER_FILE
TOTAL_MEM_MB="${SLURM_MEM_PER_NODE:-$((SLURM_CPUS_PER_TASK * 2048))}"
DEFAULT_WORKER_MEM_MB=$(( TOTAL_MEM_MB * 85 / 100 / WORKERS ))
MEM_LIMIT="${HRHSA_RSF_FIXED_WORKER_MEMORY:-${DEFAULT_WORKER_MEM_MB}MiB}"
export MEM_LIMIT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== hrHSA cm4_inter fixed-size Bayesian RSF outer diagnostic ==="
echo "workers=$WORKERS cores_per_fit=$CORES_PER_FIT models=$MODELS rows/model=$ROWS individuals/model=$INDIVIDUALS"

srun \
    --exact \
    --nodes=1 \
    --ntasks="$WORKERS" \
    --cpus-per-task="$CORES_PER_FIT" \
    --distribution=block:block \
    --cpu-bind=cores \
    bash -lc 'args=(); if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then args=(--interface "$HRHSA_DASK_INTERFACE"); fi; exec dask worker --scheduler-file "$SCHEDULER_FILE" --nthreads 1 --memory-limit "$MEM_LIMIT" --local-directory "$TMPDIR/dask-inter-rsf-fixed-${SLURM_PROCID}" "${args[@]}"' &
WORKER_STEP_PID=$!

python benchmarking/scripts/01_python/benchmark_bayesian_rsf_outer.py \
    --output "$OUTPUT" \
    --scheduler-file "$SCHEDULER_FILE" \
    --workers "$WORKERS" \
    --cores-per-fit "$CORES_PER_FIT" \
    --models "$MODELS" \
    --rows "$ROWS" \
    --predictors "$PREDICTORS" \
    --individuals "$INDIVIDUALS" \
    --sampler nutpie \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --chains "$CHAINS" \
    --target-accept "$TARGET_ACCEPT" \
    --data-seed "$SEED" \
    --sampling-seed $((10000 + SEED)) \
    --memory-monitor-interval 5 \
    --worker-startup-timeout 180

kill "$WORKER_STEP_PID" 2>/dev/null || true
wait "$WORKER_STEP_PID" 2>/dev/null || true
WORKER_STEP_PID=""
echo "cm4_inter fixed-size RSF result: $OUTPUT"
