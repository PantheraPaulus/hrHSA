#!/usr/bin/env bash
# End-to-end RSF/SSF/iSSF benchmark on the calibrated 12-core workstation.
#
# Usage:
#   bash benchmarking/scripts/03_sh/run_workstation_end_to_end.sh \
#       /tmp/surface-capacity-v1 \
#       /tmp/hrhsa-e2e-workstation [repeats]

set -euo pipefail

ROOT="${1:?Usage: $0 PREPARED_ROOT [RESULT_ROOT] [REPEATS]}"
RESULT_ROOT="${2:-/tmp/hrhsa-e2e-workstation}"
REPEATS="${3:-1}"

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEATS must be a positive integer" >&2
    exit 2
fi

OUT="$RESULT_ROOT/results.jsonl"
WORK="$RESULT_ROOT/work"
mkdir -p "$RESULT_ROOT" "$WORK"
rm -f "$OUT"

# Keep one workflow per Python process. This avoids cross-backend state leakage
# between the BlackJAX RSF run and nutpie choice-model runs and gives each E2E
# measurement an independent process-tree memory baseline.
run_one() {
    local analysis="$1"
    local inference="$2"
    local repeat="$3"
    local workspace="$WORK/${analysis}-${inference}-r${repeat}"

    echo
    echo "================================================================="
    echo "E2E ${analysis^^} / ${inference} / repeat ${repeat}"
    echo "================================================================="

    python benchmarking/scripts/01_python/benchmark_end_to_end_workflows.py \
        --root "$ROOT" \
        --output "$OUT" \
        --workspace "$workspace" \
        --analysis "$analysis" \
        --inference "$inference" \
        --repeat "$repeat"
}

for repeat in $(seq 1 "$REPEATS"); do
    # Alternate row order across repeats to reduce simple temperature/cache drift.
    if (( repeat % 2 == 1 )); then
        ORDER=(
            "rsf frequentist"
            "rsf bayesian"
            "ssf frequentist"
            "ssf bayesian"
            "issf frequentist"
            "issf bayesian"
        )
    else
        ORDER=(
            "issf bayesian"
            "issf frequentist"
            "ssf bayesian"
            "ssf frequentist"
            "rsf bayesian"
            "rsf frequentist"
        )
    fi

    for item in "${ORDER[@]}"; do
        read -r analysis inference <<< "$item"
        run_one "$analysis" "$inference" "$repeat"
    done
done

echo
echo "Campaign complete: $OUT"
python benchmarking/scripts/01_python/plot_end_to_end_workflows.py \
    --input "$OUT" \
    --output "$RESULT_ROOT/end-to-end-runtime.png"

echo "Figure: $RESULT_ROOT/end-to-end-runtime.png"
