#!/bin/bash
#SBATCH --job-name=hbayes112
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=04:00:00
#SBATCH --output=hbayes112.%j.%N.out
#SBATCH --error=hbayes112.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# Fixed-capacity Bayesian RSF geometry benchmark. Hold the scientific workload
# constant while moving the same 112 physical cores between outer-fold and
# within-fit chain concurrency:
#
#   112 folds, 28 workers x 4 cores/fit
#   112 folds, 56 workers x 2 cores/fit
#   112 folds, 112 workers x 1 core/fit
WORKERS="${1:-28}"
CORES_PER_FIT="${2:-4}"
FOLDS="${3:-112}"
SEED="${4:-42}"
CHAINS="${HRHSA_CV_CHAINS:-4}"
ROWS_PER_INDIVIDUAL="${HRHSA_CV_ROWS_PER_INDIVIDUAL:-500}"
PREDICTORS="${HRHSA_CV_PREDICTORS:-3}"
REPEATS="${HRHSA_CV_REPEATS:-3}"

if (( WORKERS <= 0 || CORES_PER_FIT <= 0 || FOLDS <= 0 )); then
    echo "WORKERS, CORES_PER_FIT and FOLDS must be positive." >&2
    exit 2
fi
if (( WORKERS * CORES_PER_FIT != SLURM_CPUS_PER_TASK )); then
    echo "Geometry ${WORKERS}x${CORES_PER_FIT} does not fill the ${SLURM_CPUS_PER_TASK}-core allocation." >&2
    exit 2
fi
if (( FOLDS < WORKERS )); then
    echo "Folds ($FOLDS) must be >= workers ($WORKERS)." >&2
    exit 2
fi
if (( CORES_PER_FIT > CHAINS )); then
    echo "cores_per_fit ($CORES_PER_FIT) must not exceed chains ($CHAINS)." >&2
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

SCHEDULER_FILE="$HRHSA_RESULT_ROOT/bayesian-outer-scheduler-${SLURM_JOB_ID}.json"
CACHE_ROOT="$HRHSA_RESULT_ROOT/bayesian-outer-cache"
OUTPUT="$HRHSA_RESULT_ROOT/bayesian-rsf-geometry-f${FOLDS}-w${WORKERS}x${CORES_PER_FIT}-${SLURM_JOB_ID}.jsonl"
rm -f "$SCHEDULER_FILE" "$OUTPUT"
mkdir -p "$CACHE_ROOT"

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

export SCHEDULER_FILE
TOTAL_MEM_MB="${SLURM_MEM_PER_NODE:-229376}"
DEFAULT_WORKER_MEM_MB=$(( TOTAL_MEM_MB * 85 / 100 / WORKERS ))
MEM_LIMIT="${HRHSA_CV_WORKER_MEMORY:-${DEFAULT_WORKER_MEM_MB}MiB}"
export MEM_LIMIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cat <<EOF
=== hrHSA Bayesian fixed-capacity CM4 geometry ===
folds=$FOLDS
sampler=nutpie
chains=$CHAINS
outer_workers=$WORKERS
cores_per_fit=$CORES_PER_FIT
worker_memory=$MEM_LIMIT
EOF

srun \
    --exact \
    --nodes=1 \
    --ntasks="$WORKERS" \
    --cpus-per-task="$CORES_PER_FIT" \
    --distribution=block:block \
    --cpu-bind=cores \
    bash -lc 'args=(); if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then args=(--interface "$HRHSA_DASK_INTERFACE"); fi; exec dask worker --scheduler-file "$SCHEDULER_FILE" --nthreads 1 --memory-limit "$MEM_LIMIT" --local-directory "$TMPDIR/dask-bayes-${SLURM_PROCID}" "${args[@]}"' &
WORKER_STEP_PID=$!

python benchmarking/scripts/01_python/benchmark_cv_scaling.py bayesian \
    --output "$OUTPUT" \
    --work-dir "$CACHE_ROOT" \
    --scheduler-file "$SCHEDULER_FILE" \
    --workers "$WORKERS" \
    --folds "$FOLDS" \
    --rows-per-individual "$ROWS_PER_INDIVIDUAL" \
    --predictors "$PREDICTORS" \
    --repeats "$REPEATS" \
    --seed "$SEED" \
    --worker-startup-timeout 600 \
    --sampler nutpie \
    --bin-width "${HRHSA_CV_BIN_WIDTH:-0.5}" \
    --random-slopes "${HRHSA_CV_RANDOM_SLOPES:-1}" \
    --draws "${HRHSA_CV_DRAWS:-250}" \
    --tune "${HRHSA_CV_TUNE:-250}" \
    --chains "$CHAINS" \
    --cores "$CORES_PER_FIT" \
    --target-accept "${HRHSA_CV_TARGET_ACCEPT:-0.9}"

kill "$WORKER_STEP_PID" 2>/dev/null || true
wait "$WORKER_STEP_PID" 2>/dev/null || true
WORKER_STEP_PID=""

echo "Bayesian fixed-capacity geometry results: $OUTPUT"
