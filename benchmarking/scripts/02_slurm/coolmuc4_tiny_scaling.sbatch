#!/bin/bash
#SBATCH --job-name=hscale28
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --hint=nomultithread
#SBATCH --mem=56G
#SBATCH --time=00:20:00
#SBATCH --output=hscale28.%j.%N.out
#SBATCH --error=hscale28.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

WORKERS="${1:-$SLURM_CPUS_PER_TASK}"
if [[ "$WORKERS" -ne "$SLURM_CPUS_PER_TASK" ]]; then
    echo "Requested worker count ($WORKERS) must equal allocated cpus-per-task ($SLURM_CPUS_PER_TASK)." >&2
    exit 2
fi
if (( WORKERS < 17 || WORKERS > 112 )); then
    echo "cm4_tiny currently permits 17-112 physical cores per job; got $WORKERS." >&2
    exit 2
fi

OUTPUT="$HRHSA_RESULT_ROOT/tiny-w${WORKERS}-${SLURM_JOB_ID}.jsonl"

# The LRZ 28-worker pilot observed transient worker RSS peaks around 1.95 GiB.
# A 1500 MB Dask nanny limit was therefore too restrictive even though the
# operation-local concurrent worker RSS remained far below the Slurm allocation.
# 2300 MB keeps spill/pause/terminate thresholds above those observed transient
# peaks while the Slurm allocation itself remains capped at 2 GiB per physical
# core by the submission helper.
python benchmarking/scripts/01_python/run_scaling.py \
    --root "$HRHSA_DATA_ROOT" \
    --output "$OUTPUT" \
    --workers "$WORKERS" \
    --benchmarks sampling,surface \
    --repeats 3 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --min-tasks-per-worker 8 \
    --backend local \
    --local-worker-memory 2300MB

echo "Results: $OUTPUT"
