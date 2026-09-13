#!/bin/bash
#SBATCH --job-name=hsmoke8
#SBATCH -D ./
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=hsmoke8.%j.%N.out
#SBATCH --error=hsmoke8.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

# LRZ reserves cm4_tiny/cm4_std for jobs using more than 16 physical cores.
# This eight-core job is therefore a serial_std functional smoke test only and
# must not be mixed into the Sapphire-Rapids CoolMUC-4 scaling curve.
OUTPUT="$HRHSA_RESULT_ROOT/serial-smoke8-${SLURM_JOB_ID}.jsonl"

python benchmarking/scripts/01_python/run_scaling.py \
    --root "$HRHSA_DATA_ROOT" \
    --output "$OUTPUT" \
    --workers 8 \
    --benchmarks sampling,surface \
    --repeats 1 \
    --spatial-chunk 1024 \
    --chunk-mb 256 \
    --min-tasks-per-worker 8 \
    --backend local \
    --local-worker-memory 4GB

echo "Smoke results: $OUTPUT"
