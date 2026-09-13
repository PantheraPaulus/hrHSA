#!/bin/bash
#SBATCH --job-name=hcv28
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --hint=nomultithread
#SBATCH --mem=56G
#SBATCH --time=00:30:00
#SBATCH --output=hcv28.%j.%N.out
#SBATCH --error=hcv28.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# One Slurm job measures exactly one outer-concurrency point.  For Bayesian CV
# the canonical production candidate is nutpie with one core per fold; this
# script therefore measures outer strong scaling (28x1, 56x1, 112x1) without
# changing within-fit chain geometry.
MODE="${1:-frequentist}"
if [[ "$MODE" == "bayesian" ]]; then
    SAMPLER="${2:-nutpie}"
else
    SAMPLER="${2:-none}"
fi
WORKERS="$SLURM_CPUS_PER_TASK"
FOLDS="${3:-112}"
SEED="${4:-42}"

if [[ "$MODE" != "frequentist" && "$MODE" != "bayesian" ]]; then
    echo "Mode must be frequentist or bayesian" >&2
    exit 2
fi
if (( WORKERS < 17 || WORKERS > 112 )); then
    echo "cm4_tiny currently permits 17-112 physical cores per job; got $WORKERS." >&2
    exit 2
fi
if (( FOLDS < WORKERS )); then
    echo "Folds ($FOLDS) must be >= workers ($WORKERS)." >&2
    exit 2
fi

if [[ "$MODE" == "frequentist" ]]; then
    ROWS_PER_INDIVIDUAL="${HRHSA_CV_ROWS_PER_INDIVIDUAL:-5000}"
    PREDICTORS="${HRHSA_CV_PREDICTORS:-6}"
    REPEATS="${HRHSA_CV_REPEATS:-3}"
else
    ROWS_PER_INDIVIDUAL="${HRHSA_CV_ROWS_PER_INDIVIDUAL:-500}"
    PREDICTORS="${HRHSA_CV_PREDICTORS:-3}"
    REPEATS="${HRHSA_CV_REPEATS:-3}"
    export HRHSA_CV_SAMPLER="$SAMPLER"
    python - <<'PY'
import importlib.util
import os
import pymc as pm
sampler = os.environ["HRHSA_CV_SAMPLER"]
if sampler != "pymc" and importlib.util.find_spec(sampler) is None:
    raise SystemExit(
        f"Requested sampler {sampler!r} is not installed. "
        "For the production Bayesian CPU campaign install .[bayesian-hpc]."
    )
print("PyMC:", pm.__version__, "sampler:", sampler)
PY
fi

SCHEDULER_FILE="$HRHSA_RESULT_ROOT/cv-scheduler-${SLURM_JOB_ID}.json"
CACHE_ROOT="$HRHSA_RESULT_ROOT/cv-scaling-cache"
OUTPUT="$HRHSA_RESULT_ROOT/cv-${MODE}-${SAMPLER}-w${WORKERS}-${SLURM_JOB_ID}.jsonl"
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
MEM_LIMIT="${HRHSA_CV_WORKER_MEMORY:-1.75GiB}"
export MEM_LIMIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== hrHSA CV outer strong-scaling campaign ==="
echo "mode=$MODE sampler=$SAMPLER workers=$WORKERS folds=$FOLDS"

echo "Starting $WORKERS one-thread/one-core Dask workers; memory=${MEM_LIMIT}/worker"
srun \
    --exact \
    --nodes=1 \
    --ntasks="$WORKERS" \
    --cpus-per-task=1 \
    --distribution=block:block \
    --cpu-bind=cores \
    bash -lc 'args=(); if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then args=(--interface "$HRHSA_DASK_INTERFACE"); fi; exec dask worker --scheduler-file "$SCHEDULER_FILE" --nthreads 1 --memory-limit "$MEM_LIMIT" --local-directory "$TMPDIR/dask-cv-${SLURM_PROCID}" "${args[@]}"' &
WORKER_STEP_PID=$!

COMMON_ARGS=(
    --output "$OUTPUT"
    --work-dir "$CACHE_ROOT"
    --scheduler-file "$SCHEDULER_FILE"
    --workers "$WORKERS"
    --folds "$FOLDS"
    --rows-per-individual "$ROWS_PER_INDIVIDUAL"
    --predictors "$PREDICTORS"
    --repeats "$REPEATS"
    --seed "$SEED"
    --worker-startup-timeout 600
)

if [[ "$MODE" == "frequentist" ]]; then
    python benchmarking/scripts/01_python/benchmark_cv_scaling.py frequentist \
        "${COMMON_ARGS[@]}" \
        --method "${HRHSA_CV_METHOD:-lbfgs}" \
        --maxiter "${HRHSA_CV_MAXITER:-200}" \
        --blas-threads 1
else
    python benchmarking/scripts/01_python/benchmark_cv_scaling.py bayesian \
        "${COMMON_ARGS[@]}" \
        --sampler "$SAMPLER" \
        --bin-width "${HRHSA_CV_BIN_WIDTH:-0.5}" \
        --random-slopes "${HRHSA_CV_RANDOM_SLOPES:-1}" \
        --draws "${HRHSA_CV_DRAWS:-250}" \
        --tune "${HRHSA_CV_TUNE:-250}" \
        --chains "${HRHSA_CV_CHAINS:-4}" \
        --cores 1 \
        --target-accept "${HRHSA_CV_TARGET_ACCEPT:-0.9}"
fi

kill "$WORKER_STEP_PID" 2>/dev/null || true
wait "$WORKER_STEP_PID" 2>/dev/null || true
WORKER_STEP_PID=""

echo "CV scaling results: $OUTPUT"
