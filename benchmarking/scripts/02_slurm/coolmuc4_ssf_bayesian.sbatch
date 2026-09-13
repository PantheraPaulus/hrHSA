#!/bin/bash
#SBATCH --job-name=hssfbayes
#SBATCH -D ./
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=hssfbayes.%j.%N.out
#SBATCH --error=hssfbayes.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

OUTPUT_DIR="${1:-$HRHSA_RESULT_ROOT/ssf-bayesian-${SLURM_JOB_ID}}"
STRATA="${2:-10000,50000}"
DRAWS="${3:-1000}"
TUNE="${4:-1000}"
mkdir -p "$OUTPUT_DIR"

for ANALYSIS in ssf issf; do
    PREDICTORS=6
    if [[ "$ANALYSIS" == "issf" ]]; then
        PREDICTORS=10
    fi

    python benchmarking/scripts/01_python/benchmark_ssf_inference.py bayesian \
        --output "$OUTPUT_DIR/${ANALYSIS}.jsonl" \
        --analysis "$ANALYSIS" \
        --strata "$STRATA" \
        --choices 11 \
        --predictors "$PREDICTORS" \
        --individuals 20 \
        --samplers pymc,blackjax \
        --draws "$DRAWS" \
        --tune "$TUNE" \
        --chains 4 \
        --cores 4 \
        --target-accept 0.9 \
        --repeats 3

done

echo "Bayesian SSF/iSSF inference results: $OUTPUT_DIR"
