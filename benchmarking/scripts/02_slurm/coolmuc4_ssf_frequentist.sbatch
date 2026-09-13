#!/bin/bash
#SBATCH --job-name=hssffreq
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=hssffreq.%j.%N.out
#SBATCH --error=hssffreq.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

OUTPUT_DIR="${1:-$HRHSA_RESULT_ROOT/ssf-inference-${SLURM_JOB_ID}}"
STRATA="${2:-10000,100000,500000}"
CHOICES="${3:-11}"
INDIVIDUALS="${4:-20}"
mkdir -p "$OUTPUT_DIR"

for ANALYSIS in ssf issf; do
    PREDICTORS=6
    if [[ "$ANALYSIS" == "issf" ]]; then
        PREDICTORS=14
    fi

    for METHOD in bfgs lbfgs; do
        OUT="$OUTPUT_DIR/${ANALYSIS}-${METHOD}.jsonl"
        python benchmarking/scripts/01_python/benchmark_ssf_inference.py frequentist \
            --output "$OUT" \
            --analysis "$ANALYSIS" \
            --strata "$STRATA" \
            --choices "$CHOICES" \
            --predictors "$PREDICTORS" \
            --individuals "$INDIVIDUALS" \
            --engines fast,statsmodels \
            --statsmodels-max-strata 50000 \
            --method "$METHOD" \
            --maxiter 300 \
            --blas-threads 1,2,4,8,16,32 \
            --objective-evaluations 20 \
            --per-id-workers 1,2,4,8,16,20 \
            --per-id-executors thread,process \
            --per-id-method lbfgs \
            --per-id-max-strata 100000 \
            --repeats 3
    done
done

echo "Frequentist SSF/iSSF inference results: $OUTPUT_DIR"
