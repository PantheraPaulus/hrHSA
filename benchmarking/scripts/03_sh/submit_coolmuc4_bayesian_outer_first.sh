#!/bin/bash
# Submit the fixed-112-core nutpie geometry campaign. Jobs are serialized so the
# campaign occupies at most one full CM4 node at a time. The scientific workload
# is held constant at 112 folds for every geometry.
#
# Usage:
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_bayesian_outer_first.sh
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_bayesian_outer_first.sh 42

set -euo pipefail

SEED="${1:-42}"
TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_bayesian_outer_first.sbatch"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ROOT="${HRHSA_SUBMISSION_ROOT:-${SCRATCH_DSS:-$PWD}/hrhsa-benchmark-submissions}"
GEN_DIR="$SUBMISSION_ROOT/bayesian-outer-first-$STAMP"
mkdir -p "$GEN_DIR"

previous_job=""
for geometry in "28 4" "56 2" "112 1"; do
    read -r workers cores_per_fit <<<"$geometry"
    short_name="hb${workers}"
    script="$GEN_DIR/${short_name}.sbatch"

    sed -e "s/hbayes112/${short_name}/g" "$TEMPLATE" >"$script"

    if [[ -n "$previous_job" ]]; then
        sed -i "/^#SBATCH --export=NONE$/a #SBATCH --dependency=afterok:${previous_job}" "$script"
    fi

    echo "Submitting $script: folds=112 geometry=${workers}x${cores_per_fit} seed=$SEED"
    submission=$(sbatch "$script" "$workers" "$cores_per_fit" 112 "$SEED")
    echo "$submission"
    previous_job="${submission##* }"
    if [[ ! "$previous_job" =~ ^[0-9]+$ ]]; then
        echo "Could not parse Slurm job ID from: $submission" >&2
        exit 3
    fi
done

echo "Concrete submitted scripts retained under: $GEN_DIR"
echo "Fixed-capacity geometry points are serialized through job $previous_job."
