#!/bin/bash
#SBATCH --job-name=hinfer
#SBATCH -D ./
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=hinfer.%j.%N.out
#SBATCH --error=hinfer.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

MODE="${1:-all}"
OUTPUT_DIR="${2:-$HRHSA_RESULT_ROOT/inference-${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_DIR"

run_frequentist() {
    python benchmarking/scripts/01_python/benchmark_inference.py frequentist \
        --output-dir "$OUTPUT_DIR/frequentist" \
        --rows 500000 \
        --predictors 6 \
        --individuals 20 \
        --methods newton,lbfgs,bfgs \
        --blas-threads 1,2,4,8,16,32 \
        --repeats 3
}

run_bayesian() {
    python benchmarking/scripts/01_python/benchmark_inference.py bayesian \
        --output-dir "$OUTPUT_DIR/bayesian" \
        --rows 100000 \
        --predictors 3 \
        --individuals 20 \
        --random-slopes 1 \
        --samplers pymc,nutpie,numpyro,blackjax \
        --draws 500 \
        --tune 500 \
        --chains 4 \
        --cores 4 \
        --blas-cores 1 \
        --target-accept 0.9 \
        --eta-storage both \
        --repeats 1 \
        --save-posterior-summary
}

case "$MODE" in
    frequentist)
        run_frequentist
        ;;
    bayesian)
        run_bayesian
        ;;
    all)
        run_frequentist
        run_bayesian
        ;;
    *)
        echo "Usage: sbatch coolmuc4_inference.sbatch [frequentist|bayesian|all] [OUTPUT_DIR]" >&2
        exit 2
        ;;
esac

echo "Inference benchmark results: $OUTPUT_DIR"
