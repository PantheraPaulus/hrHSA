#!/bin/bash
# Submit exactly one short Bayesian diagnostic to cm4_inter.
#
# The user's LRZ association permits only a very small number of submitted jobs,
# so this helper intentionally never loops over a campaign. Submit the next point
# only after a previous job has left the queue.
#
# Usage:
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh rsf-fixed 56x2
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh ssf 28x4
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh issf 28x4
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh ssf-capacity 10000
#   bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh issf-capacity 10000
#
# Optional third argument is the seed (default 42).

set -euo pipefail

MODE="${1:-}"
POINT="${2:-}"
SEED="${3:-42}"
if [[ -z "$MODE" || -z "$POINT" ]]; then
    echo "Usage: $0 {rsf-scaling|rsf-geometry|rsf-fixed|ssf|issf|ssf-capacity|issf-capacity} POINT [SEED]" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ROOT="${HRHSA_SUBMISSION_ROOT:-${SCRATCH_DSS:-$PWD}/hrhsa-benchmark-submissions}"
GEN_DIR="$SUBMISSION_ROOT/inter-bayesian-${MODE}-${POINT}-$STAMP"
mkdir -p "$GEN_DIR"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"

submit_one() {
    local script="$1"
    shift
    if [[ -n "$GIT_COMMIT" ]]; then
        # --export=NONE deliberately drops the submission environment. Stamp the
        # exact source SHA into the generated script so benchmark records remain
        # attributable even when git is unavailable on the compute node.
        sed -i "/^activate_hrhsa$/i export HRHSA_GIT_COMMIT=\"$GIT_COMMIT\"" "$script"
    fi
    echo "Submitting $script $*"
    sbatch "$script" "$@"
}

case "$MODE" in
    rsf-scaling)
        workers="$POINT"
        case "$workers" in
            28) wall="00:18:00" ;;
            56) wall="00:12:00" ;;
            112) wall="00:08:00" ;;
            *) echo "rsf-scaling POINT must be 28, 56 or 112" >&2; exit 2 ;;
        esac
        TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_inter_bayesian_cv.sbatch"
        memory_gb=$((workers * 2))
        short_name="hicv${workers}"
        script="$GEN_DIR/${short_name}.sbatch"
        sed \
            -e "s/hicv28/${short_name}/g" \
            -e "s/#SBATCH --cpus-per-task=28/#SBATCH --cpus-per-task=${workers}/" \
            -e "s/#SBATCH --mem=56G/#SBATCH --mem=${memory_gb}G/" \
            -e "s/#SBATCH --time=00:10:00/#SBATCH --time=${wall}/" \
            "$TEMPLATE" >"$script"
        submit_one "$script" nutpie 112 "$SEED"
        ;;

    rsf-geometry)
        workers="$POINT"
        case "$workers" in
            28) cores_per_fit=4 ;;
            56) cores_per_fit=2 ;;
            112) cores_per_fit=1 ;;
            *) echo "rsf-geometry POINT must be 28, 56 or 112" >&2; exit 2 ;;
        esac
        TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_inter_bayesian_outer_first.sbatch"
        short_name="hib${workers}"
        script="$GEN_DIR/${short_name}.sbatch"
        sed -e "s/hibayes112/${short_name}/g" "$TEMPLATE" >"$script"
        submit_one "$script" "$workers" "$cores_per_fit" 112 "$SEED"
        ;;

    rsf-fixed)
        if [[ ! "$POINT" =~ ^(28|56|112)x([124])$ ]]; then
            echo "rsf-fixed POINT must look like 28x1, 56x1, 112x1, 28x4, 56x2, etc." >&2
            exit 2
        fi
        workers="${BASH_REMATCH[1]}"
        cores_per_fit="${BASH_REMATCH[2]}"
        total_cores=$((workers * cores_per_fit))
        if (( total_cores > 112 )); then
            echo "Requested geometry uses $total_cores cores; maximum is 112." >&2
            exit 2
        fi
        TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_inter_bayesian_rsf_fixed.sbatch"
        memory_gb=$((total_cores * 2))
        short_name="hir${workers}c${cores_per_fit}"
        script="$GEN_DIR/${short_name}.sbatch"
        sed \
            -e "s/hirsf28/${short_name}/g" \
            -e "s/#SBATCH --cpus-per-task=28/#SBATCH --cpus-per-task=${total_cores}/" \
            -e "s/#SBATCH --mem=56G/#SBATCH --mem=${memory_gb}G/" \
            "$TEMPLATE" >"$script"
        submit_one "$script" "$workers" "$cores_per_fit" 112 "$SEED"
        ;;

    ssf|issf)
        if [[ ! "$POINT" =~ ^(28|56|112)x([124])$ ]]; then
            echo "$MODE POINT must look like 28x4, 56x2 or 112x1." >&2
            exit 2
        fi
        workers="${BASH_REMATCH[1]}"
        cores_per_fit="${BASH_REMATCH[2]}"
        total_cores=$((workers * cores_per_fit))
        if (( total_cores != 112 )); then
            echo "$MODE topology diagnostics currently require a full 112-core geometry; use 28x4, 56x2 or 112x1." >&2
            exit 2
        fi

        TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_inter_bayesian_choice_outer.sbatch"
        short_name="hi${MODE:0:1}${workers}c${cores_per_fit}"
        script="$GEN_DIR/${short_name}.sbatch"
        if [[ "$MODE" == "issf" ]]; then
            wall="00:10:00"
        else
            wall="00:08:00"
        fi
        sed \
            -e "s/hicho112/${short_name}/g" \
            -e "s/#SBATCH --time=00:08:00/#SBATCH --time=${wall}/" \
            "$TEMPLATE" >"$script"
        submit_one "$script" "$MODE" "$workers" "$cores_per_fit" 112 "$SEED"
        ;;

    ssf-capacity|issf-capacity)
        analysis="${MODE%-capacity}"
        strata="$POINT"
        case "$strata" in
            5000|10000|20000|50000) ;;
            *) echo "$MODE POINT must be one of 5000, 10000, 20000 or 50000 strata." >&2; exit 2 ;;
        esac

        # Capacity uses one complete full-node wave: 28 independent models at the
        # frozen 28x4 choice-model geometry. 500+500 matches the workstation
        # capacity campaign so runtime and memory scaling are directly comparable.
        workers=28
        cores_per_fit=4
        models=28
        TEMPLATE="benchmarking/scripts/02_slurm/coolmuc4_inter_bayesian_choice_outer.sbatch"

        if [[ "$analysis" == "ssf" ]]; then
            case "$strata" in
                5000) wall="00:08:00" ;;
                10000) wall="00:12:00" ;;
                20000) wall="00:20:00" ;;
                50000) wall="00:35:00" ;;
            esac
            prefix="hsc"
        else
            case "$strata" in
                5000) wall="00:10:00" ;;
                10000) wall="00:15:00" ;;
                20000) wall="00:25:00" ;;
                50000) wall="00:45:00" ;;
            esac
            prefix="hic"
        fi

        case "$strata" in
            5000) size_tag="5k" ;;
            10000) size_tag="10k" ;;
            20000) size_tag="20k" ;;
            50000) size_tag="50k" ;;
        esac
        short_name="${prefix}${size_tag}"
        script="$GEN_DIR/${short_name}.sbatch"
        sed \
            -e "s/hicho112/${short_name}/g" \
            -e "s/#SBATCH --time=00:08:00/#SBATCH --time=${wall}/" \
            -e 's/STRATA="${HRHSA_CHOICE_STRATA:-5000}"/STRATA="${HRHSA_CHOICE_STRATA:-'"$strata"'}"/' \
            -e 's/DRAWS="${HRHSA_CHOICE_DRAWS:-250}"/DRAWS="${HRHSA_CHOICE_DRAWS:-500}"/' \
            -e 's/TUNE="${HRHSA_CHOICE_TUNE:-250}"/TUNE="${HRHSA_CHOICE_TUNE:-500}"/' \
            "$TEMPLATE" >"$script"
        submit_one "$script" "$analysis" "$workers" "$cores_per_fit" "$models" "$SEED"
        ;;

    *)
        echo "MODE must be rsf-scaling, rsf-geometry, rsf-fixed, ssf, issf, ssf-capacity, or issf-capacity." >&2
        exit 2
        ;;
esac

echo "Concrete submitted script retained under: $GEN_DIR"
echo "Exactly one job was submitted. Submit the next point only after queue capacity is available."
