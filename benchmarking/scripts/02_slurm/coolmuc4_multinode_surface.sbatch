#!/bin/bash
#SBATCH --job-name=hmulti
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=hmulti.%j.%N.out
#SBATCH --error=hmulti.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

NODES="$SLURM_JOB_NUM_NODES"
if (( NODES < 2 || NODES > 4 )); then
    echo "cm4_std currently permits 2-4 nodes per job; got $NODES." >&2
    exit 2
fi

WORKERS=$((NODES * 112 - 1))
SCHEDULER_FILE="$HRHSA_RESULT_ROOT/scheduler-${SLURM_JOB_ID}.json"
OUTPUT="$HRHSA_RESULT_ROOT/multinode-n${NODES}-w${WORKERS}-${SLURM_JOB_ID}.jsonl"
rm -f "$SCHEDULER_FILE"
export SCHEDULER_FILE

INTERFACE_ARGS=()
if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then
    INTERFACE_ARGS=(--interface "$HRHSA_DASK_INTERFACE")
fi

mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
FIRST_NODE="${HOSTS[0]}"

SCHED_PID=""
FIRST_WORKER_PID=""
OTHER_WORKER_PID=""
cleanup() {
    set +e
    [[ -n "$OTHER_WORKER_PID" ]] && kill "$OTHER_WORKER_PID" 2>/dev/null
    [[ -n "$FIRST_WORKER_PID" ]] && kill "$FIRST_WORKER_PID" 2>/dev/null
    [[ -n "$SCHED_PID" ]] && kill "$SCHED_PID" 2>/dev/null
    rm -f "$SCHEDULER_FILE"
}
trap cleanup EXIT

# One physical CPU on the first node is reserved for the scheduler. Every other
# allocated physical CPU is occupied by one Dask worker.
srun \
    --exact \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --nodelist="$FIRST_NODE" \
    --cpu-bind=cores \
    --exclusive \
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

WORKER_MEMORY="${HRHSA_MULTINODE_WORKER_MEMORY:-1.5GB}"
export WORKER_MEMORY

srun \
    --exact \
    --nodes=1 \
    --ntasks=111 \
    --cpus-per-task=1 \
    --nodelist="$FIRST_NODE" \
    --cpu-bind=cores \
    --exclusive \
    bash -lc 'args=(); if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then args=(--interface "$HRHSA_DASK_INTERFACE"); fi; exec dask worker --scheduler-file "$SCHEDULER_FILE" --nthreads 1 --memory-limit "$WORKER_MEMORY" --local-directory "$TMPDIR/dask-${SLURM_PROCID}" "${args[@]}"' &
FIRST_WORKER_PID=$!

OTHER_COUNT=$((NODES - 1))
OTHER_WORKERS=$((OTHER_COUNT * 112))
OTHER_NODELIST=$(IFS=,; echo "${HOSTS[*]:1}")
srun \
    --exact \
    --nodes="$OTHER_COUNT" \
    --ntasks="$OTHER_WORKERS" \
    --ntasks-per-node=112 \
    --cpus-per-task=1 \
    --nodelist="$OTHER_NODELIST" \
    --distribution=block:block \
    --cpu-bind=cores \
    --exclusive \
    bash -lc 'args=(); if [[ -n "${HRHSA_DASK_INTERFACE:-}" ]]; then args=(--interface "$HRHSA_DASK_INTERFACE"); fi; exec dask worker --scheduler-file "$SCHEDULER_FILE" --nthreads 1 --memory-limit "$WORKER_MEMORY" --local-directory "$TMPDIR/dask-${SLURM_PROCID}" "${args[@]}"' &
OTHER_WORKER_PID=$!

python benchmarking/scripts/01_python/run_scaling.py \
    --root "$HRHSA_LARGE_DATA_ROOT" \
    --output "$OUTPUT" \
    --workers "$WORKERS" \
    --benchmarks surface \
    --repeats 3 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --min-tasks-per-worker 8 \
    --backend scheduler-file \
    --scheduler-file "$SCHEDULER_FILE" \
    --worker-startup-timeout 1800

echo "Results: $OUTPUT"
