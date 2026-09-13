#!/bin/bash
#SBATCH --job-name=hchoice112
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
#SBATCH --output=hchoice112.%j.%N.out
#SBATCH --error=hchoice112.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

ANALYSIS="${1:-ssf}"
MODELS="${2:-112}"
SEED="${3:-42}"
CHAINS="${HRHSA_CHOICE_CHAINS:-4}"
STRATA="${HRHSA_CHOICE_STRATA:-10000}"
CHOICES="${HRHSA_CHOICE_CHOICES:-11}"
INDIVIDUALS="${HRHSA_CHOICE_INDIVIDUALS:-20}"
DRAWS="${HRHSA_CHOICE_DRAWS:-500}"
TUNE="${HRHSA_CHOICE_TUNE:-500}"
TARGET_ACCEPT="${HRHSA_CHOICE_TARGET_ACCEPT:-0.9}"

if [[ "$ANALYSIS" != "ssf" && "$ANALYSIS" != "issf" ]]; then
    echo "ANALYSIS must be ssf or issf." >&2
    exit 2
fi
if (( MODELS <= 0 )); then
    echo "MODELS must be positive." >&2
    exit 2
fi

if [[ "$ANALYSIS" == "ssf" ]]; then
    PREDICTORS="${HRHSA_CHOICE_PREDICTORS:-6}"
else
    PREDICTORS="${HRHSA_CHOICE_PREDICTORS:-14}"
fi

PLAN=$(python benchmarking/scripts/01_python/plan_bayesian_execution.py \
    --analysis "$ANALYSIS" \
    --models "$MODELS" \
    --available-cores "$SLURM_CPUS_PER_TASK" \
    --chains "$CHAINS" \
    --objective throughput \
    --profile coolmuc4 \
    --format shell)
eval "$PLAN"

WORKERS="$HRHSA_BAYES_OUTER_WORKERS"
CORES_PER_FIT="$HRHSA_BAYES_CORES_PER_FIT"
SAMPLER="$HRHSA_BAYES_SAMPLER"
ACTIVE_CORES="$HRHSA_BAYES_ACTIVE_CORES"
UNUSED_CORES="$HRHSA_BAYES_UNUSED_CORES"

if [[ "$SAMPLER" != "nutpie" ]]; then
    echo "Choice-model throughput planner unexpectedly selected sampler=$SAMPLER" >&2
    exit 2
fi
if (( UNUSED_CORES > 0 )) && [[ "${HRHSA_ALLOW_IDLE_CORES:-0}" != "1" ]]; then
    echo "Measured CoolMUC-4 plan would leave $UNUSED_CORES/$SLURM_CPUS_PER_TASK cores idle for MODELS=$MODELS." >&2
    echo "Set HRHSA_ALLOW_IDLE_CORES=1 if the ensemble has fewer than 28 independent fits." >&2
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

SCHEDULER_FILE="$HRHSA_RESULT_ROOT/choice-outer-scheduler-${SLURM_JOB_ID}.json"
OUTPUT="$HRHSA_RESULT_ROOT/bayesian-${ANALYSIS}-outer-m${MODELS}-w${WORKERS}x${CORES_PER_FIT}-${SLURM_JOB_ID}.jsonl"
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
MEM_LIMIT="${HRHSA_CHOICE_WORKER_MEMORY:-${DEFAULT_WORKER_MEM_MB}MiB}"
export MEM_LIMIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cat <<EOF
=== hrHSA Bayesian ${ANALYSIS} outer-model CM4 geometry ===
models=$MODELS
strata/model=$STRATA
predictors=$PREDICTORS
sampler=$SAMPLER
profile=$HRHSA_BAYES_PROFILE
chains=$CHAINS
outer_workers=$WORKERS
cores_per_fit=$CORES_PER_FIT
active_cores=$ACTIVE_CORES
unused_cores=$UNUSED_CORES
worker_memory=$MEM_LIMIT
EOF

srun \
    --exact \
    --nodes=1 \
    --ntasks="$WORKERS" \
    --cpus-per-task="$CORES_PER_FIT" \
    --distribution=block:block \
    --cpu-bind=cores \
    bash -lc 'args=(); if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then args=(--interface "$HRHSA_DASK_INTERFACE"); fi; exec dask worker --scheduler-file "$SCHEDULER_FILE" --nthreads 1 --memory-limit "$MEM_LIMIT" --local-directory "$TMPDIR/dask-choice-${SLURM_PROCID}" "${args[@]}"' &
WORKER_STEP_PID=$!

python benchmarking/scripts/01_python/benchmark_bayesian_choice_outer.py \
    --analysis "$ANALYSIS" \
    --output "$OUTPUT" \
    --scheduler-file "$SCHEDULER_FILE" \
    --workers "$WORKERS" \
    --cores-per-fit "$CORES_PER_FIT" \
    --models "$MODELS" \
    --strata "$STRATA" \
    --choices "$CHOICES" \
    --predictors "$PREDICTORS" \
    --individuals "$INDIVIDUALS" \
    --sampler nutpie \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --chains "$CHAINS" \
    --target-accept "$TARGET_ACCEPT" \
    --data-seed "$SEED" \
    --sampling-seed $((10000 + SEED)) \
    --worker-startup-timeout 600

kill "$WORKER_STEP_PID" 2>/dev/null || true
wait "$WORKER_STEP_PID" 2>/dev/null || true
WORKER_STEP_PID=""

echo "Bayesian choice-model outer results: $OUTPUT"
