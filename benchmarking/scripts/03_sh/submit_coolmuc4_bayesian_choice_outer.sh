#!/bin/bash
# Submit fixed-112-core Bayesian SSF/iSSF outer-model geometry points.
#
# Usage:
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_bayesian_choice_outer.sh ssf
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_bayesian_choice_outer.sh issf

set -euo pipefail

ANALYSIS="${1:-ssf}"
SEED="${2:-42}"
if [[ "$ANALYSIS" != "ssf" && "$ANALYSIS" != "issf" ]]; then
    echo "ANALYSIS must be ssf or issf." >&2
    exit 2
fi

TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_bayesian_choice_outer.sbatch"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ROOT="${HRHSA_SUBMISSION_ROOT:-${SCRATCH_DSS:-$PWD}/hrhsa-benchmark-submissions}"
GEN_DIR="$SUBMISSION_ROOT/bayesian-${ANALYSIS}-outer-$STAMP"
mkdir -p "$GEN_DIR"

previous_job=""
for models in 28 56 112; do
    short_name="h${ANALYSIS:0:1}${models}"
    script="$GEN_DIR/${short_name}.sbatch"

    sed \
        -e "s/hchoice112/${short_name}/g" \
        "$TEMPLATE" >"$script"

    if [[ -n "$previous_job" ]]; then
        sed -i "/^#SBATCH --export=NONE$/a #SBATCH --dependency=afterok:${previous_job}" "$script"
    fi

    echo "Submitting $script: analysis=$ANALYSIS models=$models seed=$SEED"
    submission=$(sbatch "$script" "$ANALYSIS" "$models" "$SEED")
    echo "$submission"
    previous_job="${submission##* }"
    if [[ ! "$previous_job" =~ ^[0-9]+$ ]]; then
        echo "Could not parse Slurm job ID from: $submission" >&2
        exit 3
    fi
done

echo "Concrete submitted scripts retained under: $GEN_DIR"
echo "Choice-model geometry points are serialized through job $previous_job."
