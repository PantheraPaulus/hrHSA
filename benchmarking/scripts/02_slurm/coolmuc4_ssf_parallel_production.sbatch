#!/bin/bash
#SBATCH --job-name=hssfprod
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
#SBATCH --output=hssfprod.%j.%N.out
#SBATCH --error=hssfprod.%j.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

module load slurm_setup
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source benchmarking/scripts/03_sh/coolmuc4_common.sh
activate_hrhsa

OUTPUT_DIR="${1:-$HRHSA_RESULT_ROOT/ssf-production-per-id-${SLURM_JOB_ID}}"
STRATA="${2:-100000}"
CHOICES="${3:-11}"
INDIVIDUALS="${4:-20}"
WORKERS="${5:-1,2,4,6,8,12,16,20}"
mkdir -p "$OUTPUT_DIR"

# Outer model parallelism owns the physical cores in this campaign.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python benchmarking/scripts/01_python/benchmark_ssf_parallel_production.py \
    --output "$OUTPUT_DIR/ssf-process.jsonl" \
    --analysis ssf \
    --strata "$STRATA" \
    --choices "$CHOICES" \
    --predictors 6 \
    --individuals "$INDIVIDUALS" \
    --workers "$WORKERS" \
    --executor process \
    --native-threads 1 \
    --method lbfgs \
    --maxiter 300 \
    --repeats 3

python benchmarking/scripts/01_python/benchmark_ssf_parallel_production.py \
    --output "$OUTPUT_DIR/issf-process.jsonl" \
    --analysis issf \
    --strata "$STRATA" \
    --choices "$CHOICES" \
    --predictors 14 \
    --individuals "$INDIVIDUALS" \
    --workers "$WORKERS" \
    --executor process \
    --native-threads 1 \
    --method lbfgs \
    --maxiter 300 \
    --repeats 3

echo "Production per-individual SSF/iSSF results: $OUTPUT_DIR"
