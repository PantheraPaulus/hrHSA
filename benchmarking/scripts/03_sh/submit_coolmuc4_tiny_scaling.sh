#!/bin/bash
# Generate and submit concrete LRZ-compliant cm4_tiny job scripts. Keeping the
# resource directives in the generated batch script avoids submission-time
# overrides, which LRZ discourages because they make jobs harder to reproduce.
# Scaling points are chained with afterok dependencies so this campaign never
# contributes more than one memory-heavy job to the user's aggregate memory QoS.

set -euo pipefail

TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_tiny_scaling.sbatch"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ROOT="${HRHSA_SUBMISSION_ROOT:-${SCRATCH_DSS:-$PWD}/hrhsa-benchmark-submissions}"
GEN_DIR="$SUBMISSION_ROOT/tiny-scaling-$STAMP"
mkdir -p "$GEN_DIR"

if ! git_commit="$(git rev-parse HEAD 2>/dev/null)" || [[ -z "$git_commit" ]]; then
    echo "Could not determine the hrHSA git commit; refusing an unversioned publication benchmark." >&2
    exit 2
fi

echo "Benchmark source commit: $git_commit"
previous_job=""

# All points satisfy the current cm4_tiny 17-112 physical-core range. Memory is
# capped at 2 GiB/core, so even 112 cores request only 224 GiB.
for workers in 28 56 84 112; do
    memory_gb=$((workers * 2))
    short_name="hs${workers}"
    script="$GEN_DIR/${short_name}.sbatch"

    sed \
        -e "s/hscale28/${short_name}/g" \
        -e "s/#SBATCH --cpus-per-task=28/#SBATCH --cpus-per-task=${workers}/" \
        -e "s/#SBATCH --mem=56G/#SBATCH --mem=${memory_gb}G/" \
        "$TEMPLATE" >"$script"

    # Pin provenance at submission time rather than relying on git being present
    # on a compute node. Keep LRZ's required slurm_setup line first executable.
    sed -i "/^set -euo pipefail$/a export HRHSA_GIT_COMMIT=\"${git_commit}\"" "$script"

    if [[ -n "$previous_job" ]]; then
        sed -i "/^#SBATCH --export=NONE$/a #SBATCH --dependency=afterok:${previous_job}" "$script"
    fi

    echo "Submitting $script: workers=$workers memory=${memory_gb}G commit=${git_commit:0:12}"
    submission=$(sbatch "$script" "$workers")
    echo "$submission"
    if [[ "$submission" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
        previous_job="${BASH_REMATCH[1]}"
    else
        echo "Could not parse Slurm job ID from: $submission" >&2
        exit 3
    fi
done

echo "Concrete submitted scripts retained under: $GEN_DIR"
echo "Scaling points are serialized through job $previous_job."
