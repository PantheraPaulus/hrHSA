#!/bin/bash
#SBATCH --job-name=hicho112
#SBATCH -D ./
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --hint=nomultithread
#SBATCH --mem=224G
#SBATCH --time=00:08:00
#SBATCH --output=hicho112.%j.%N.out
#SBATCH --error=hicho112.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# Short cm4_inter fixed-size diagnostic for hierarchical SSF/iSSF fits.
# Hold the scientific workload fixed (112 independent 5k-stratum models) and
# vary only the placement of the 112 physical CPU cores between outer Dask fits
# and nutpie chain concurrency. These are topology diagnostics, not posterior
# validation or capacity runs: workstation validation retains the longer 500+500
# campaigns and the workstation capacity curve covers 2.5k--50k strata.
ANALYSIS="${1:-ssf}"
WORKERS="${2:-56}"
CORES_PER_FIT="${3:-2}"
MODELS="${4:-112}"
SEED="${5:-42}"
CHAINS="${HRHSA_CHOICE_CHAINS:-4}"
STRATA="${HRHSA_CHOICE_STRATA:-5000}"
CHOICES="${HRHSA_CHOICE_CHOICES:-11}"
INDIVIDUALS="${HRHSA_CHOICE_INDIVIDUALS:-20}"
DRAWS="${HRHSA_CHOICE_DRAWS:-250}"
TUNE="${HRHSA_CHOICE_TUNE:-250}"
TARGET_ACCEPT="${HRHSA_CHOICE_TARGET_ACCEPT:-0.9}"

if [[ "$ANALYSIS" != "ssf" && "$ANALYSIS" != "issf" ]]; then
    echo "ANALYSIS must be ssf or issf." >&2
    exit 2
fi
if (( WORKERS <= 0 || CORES_PER_FIT <= 0 || MODELS < WORKERS )); then
    echo "Require WORKERS>0, CORES_PER_FIT>0 and MODELS>=WORKERS." >&2
    exit 2
fi
ACTIVE_CORES=$(( WORKERS * CORES_PER_FIT ))
if (( ACTIVE_CORES > SLURM_CPUS_PER_TASK )); then
    echo "Requested geometry uses $ACTIVE_CORES cores; allocation has $SLURM_CPUS_PER_TASK." >&2
    exit 2
fi
if (( CORES_PER_FIT > CHAINS )); then
    echo "CORES_PER_FIT ($CORES_PER_FIT) cannot exceed chains ($CHAINS)." >&2
    exit 2
fi

if [[ "$ANALYSIS" == "ssf" ]]; then
    PREDICTORS="${HRHSA_CHOICE_PREDICTORS:-6}"
else
    PREDICTORS="${HRHSA_CHOICE_PREDICTORS:-14}"
fi

python - <<'PY'
import importlib.util
import pymc as pm
if importlib.util.find_spec("nutpie") is None:
    raise SystemExit("nutpie is required; install the project with .[bayesian-hpc]")
import nutpie
print("PyMC:", pm.__version__, "nutpie:", nutpie.__version__)
PY

SCHEDULER_FILE="$HRHSA_RESULT_ROOT/inter-choice-outer-scheduler-${SLURM_JOB_ID}.json"
OUTPUT="$HRHSA_RESULT_ROOT/inter-bayesian-${ANALYSIS}-fixed-w${WORKERS}x${CORES_PER_FIT}-${SLURM_JOB_ID}.jsonl"
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
TOTAL_MEM_MB="${SLURM_MEM_PER_NODE:-229376}"
DEFAULT_WORKER_MEM_MB=$(( TOTAL_MEM_MB * 85 / 100 / WORKERS ))
MEM_LIMIT="${HRHSA_CHOICE_WORKER_MEMORY:-${DEFAULT_WORKER_MEM_MB}MiB}"
export MEM_LIMIT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== hrHSA cm4_inter Bayesian ${ANALYSIS} fixed-size outer diagnostic ==="
echo "models=$MODELS workers=$WORKERS cores_per_fit=$CORES_PER_FIT active_cores=$ACTIVE_CORES strata=$STRATA choices=$CHOICES predictors=$PREDICTORS draws=$DRAWS tune=$TUNE"

srun \
    --exact \
    --nodes=1 \
    --ntasks="$WORKERS" \
    --cpus-per-task="$CORES_PER_FIT" \
    --distribution=block:block \
    --cpu-bind=cores \
    bash -lc 'args=(); if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then args=(--interface "$HRHSA_DASK_INTERFACE"); fi; exec dask worker --scheduler-file "$SCHEDULER_FILE" --nthreads 1 --memory-limit "$MEM_LIMIT" --local-directory "$TMPDIR/dask-inter-choice-${SLURM_PROCID}" "${args[@]}"' &
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
    --worker-startup-timeout 300

kill "$WORKER_STEP_PID" 2>/dev/null || true
wait "$WORKER_STEP_PID" 2>/dev/null || true
WORKER_STEP_PID=""
echo "cm4_inter Bayesian choice-model result: $OUTPUT"
