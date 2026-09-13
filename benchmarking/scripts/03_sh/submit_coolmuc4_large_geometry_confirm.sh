#!/bin/bash
# Submit the three confirmatory large-raster geometries as separate serialized
# jobs. A fresh worker pool plus one untimed warm-up per job reduces the strong
# first-repeat effect observed in the discovery sweep without overlapping large
# memory requests under LRZ's aggregate user memory QoS.

set -euo pipefail

TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_large_geometry_confirm.sbatch"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ROOT="${HRHSA_SUBMISSION_ROOT:-${SCRATCH_DSS:-$PWD}/hrhsa-benchmark-submissions}"
GEN_DIR="$SUBMISSION_ROOT/large-geometry-confirm-$STAMP"
mkdir -p "$GEN_DIR"

if ! git_commit="$(git rev-parse HEAD 2>/dev/null)" || [[ -z "$git_commit" ]]; then
    echo "Could not determine the hrHSA git commit; refusing an unversioned benchmark." >&2
    exit 2
fi

echo "Benchmark source commit: $git_commit"
previous_job=""

# Deliberately non-monotonic order. Every geometry is a separate Slurm job with
# its own fresh Dask worker pool and an untimed warm-up before five timed repeats.
GEOMETRIES=(
    "16:7"
    "8:14"
    "28:4"
)

for geometry in "${GEOMETRIES[@]}"; do
    IFS=: read -r workers threads <<<"$geometry"
    script="$GEN_DIR/hgeomcf-${workers}x${threads}.sbatch"

    cp "$TEMPLATE" "$script"

    # Pin the exact source revision at submission time while retaining
    # module load slurm_setup as the first executable line.
    sed -i "/^set -euo pipefail$/a export HRHSA_GIT_COMMIT=\"${git_commit}\"" "$script"

    if [[ -n "$previous_job" ]]; then
        sed -i "/^#SBATCH --export=NONE$/a #SBATCH --dependency=afterok:${previous_job}" "$script"
    fi

    echo "Submitting geometry ${workers}x${threads} from ${git_commit:0:12}"
    submission=$(sbatch "$script" "$workers" "$threads")
    echo "$submission"

    if [[ "$submission" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
        previous_job="${BASH_REMATCH[1]}"
    else
        echo "Could not parse Slurm job ID from: $submission" >&2
        exit 3
    fi
done

echo "Concrete submitted scripts retained under: $GEN_DIR"
echo "Confirmatory geometries are serialized through job $previous_job."
