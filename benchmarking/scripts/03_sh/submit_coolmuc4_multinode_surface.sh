#!/bin/bash
# Generate and submit concrete 2- and 4-node cm4_std scripts. LRZ currently
# permits 2-4 nodes on cm4_std; the one-node reference comes from the 112-core
# cm4_tiny scaling job. The expensive node-count points are serialized.

set -euo pipefail

TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_multinode_surface.sbatch"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ROOT="${HRHSA_SUBMISSION_ROOT:-${SCRATCH_DSS:-$PWD}/hrhsa-benchmark-submissions}"
GEN_DIR="$SUBMISSION_ROOT/multinode-$STAMP"
mkdir -p "$GEN_DIR"

previous_job=""
for nodes in 2 4; do
    short_name="hm${nodes}n"
    script="$GEN_DIR/${short_name}.sbatch"
    sed \
        -e "s/hmulti/${short_name}/g" \
        -e "s/#SBATCH --nodes=2/#SBATCH --nodes=${nodes}/" \
        "$TEMPLATE" >"$script"

    if [[ -n "$previous_job" ]]; then
        sed -i "/^#SBATCH --export=NONE$/a #SBATCH --dependency=afterok:${previous_job}" "$script"
    fi

    echo "Submitting $script: nodes=$nodes"
    submission=$(sbatch "$script")
    echo "$submission"
    previous_job="${submission##* }"
    if [[ ! "$previous_job" =~ ^[0-9]+$ ]]; then
        echo "Could not parse Slurm job ID from: $submission" >&2
        exit 3
    fi
done

echo "Concrete submitted scripts retained under: $GEN_DIR"
echo "Multinode points are serialized through job $previous_job."
