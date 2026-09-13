#!/bin/bash
# Generate and submit concrete cm4_tiny CV scaling scripts without overriding
# Slurm resource directives at submission time. Worker-count points are chained
# so one campaign contributes at most one running memory-heavy job at a time.
#
# Usage:
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_cv_scaling.sh frequentist
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_cv_scaling.sh bayesian
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_cv_scaling.sh bayesian nutpie
#
# Bayesian defaults to the validated CPU-throughput candidate, nutpie.

set -euo pipefail

MODE="${1:-frequentist}"
if [[ "$MODE" == "bayesian" ]]; then
    SAMPLER="${2:-nutpie}"
else
    SAMPLER="${2:-none}"
fi
FOLDS="${3:-112}"
SEED="${4:-42}"

if [[ "$MODE" != "frequentist" && "$MODE" != "bayesian" ]]; then
    echo "Mode must be frequentist or bayesian" >&2
    exit 2
fi

TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_cv_scaling.sbatch"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ROOT="${HRHSA_SUBMISSION_ROOT:-${SCRATCH_DSS:-$PWD}/hrhsa-benchmark-submissions}"
GEN_DIR="$SUBMISSION_ROOT/cv-${MODE}-${SAMPLER}-$STAMP"
mkdir -p "$GEN_DIR"

previous_job=""

# 28/56/112 all satisfy cm4_tiny's current >=17-core rule and divide the default
# 112-fold campaign exactly, so the scaling curve has no partially filled final
# scheduling wave. Bayesian points use one nutpie core per fold to isolate outer
# strong scaling; the fixed-capacity 28x4/56x2/112x1 geometry is a separate
# campaign in submit_coolmuc4_bayesian_outer_first.sh.
for workers in 28 56 112; do
    memory_gb=$((workers * 2))
    short_name="hc${workers}"
    script="$GEN_DIR/${short_name}.sbatch"

    sed \
        -e "s/hcv28/${short_name}/g" \
        -e "s/#SBATCH --cpus-per-task=28/#SBATCH --cpus-per-task=${workers}/" \
        -e "s/#SBATCH --mem=56G/#SBATCH --mem=${memory_gb}G/" \
        "$TEMPLATE" >"$script"

    if [[ -n "$previous_job" ]]; then
        sed -i "/^#SBATCH --export=NONE$/a #SBATCH --dependency=afterok:${previous_job}" "$script"
    fi

    echo "Submitting $script: mode=$MODE sampler=$SAMPLER workers=$workers folds=$FOLDS"
    submission=$(sbatch "$script" "$MODE" "$SAMPLER" "$FOLDS" "$SEED")
    echo "$submission"
    previous_job="${submission##* }"
    if [[ ! "$previous_job" =~ ^[0-9]+$ ]]; then
        echo "Could not parse Slurm job ID from: $submission" >&2
        exit 3
    fi
done

echo "Concrete submitted scripts retained under: $GEN_DIR"
echo "CV scaling points are serialized through job $previous_job."
