#!/usr/bin/env bash
# Small A/B validation of recursive point-shard bisection on the workstation.
#
# Usage:
#   bash benchmarking/scripts/03_sh/run_workstation_point_bisection_validation.sh \
#       /path/to/prepared-benchmark-root \
#       /tmp/hrhsa-point-bisection-validation
#
# The campaign keeps the calibrated workstation setup fixed at 6x2, 24 GiB,
# 100M points and compares the unchanged direct partitioned sampler against six
# synchronized 1x2 spatial shards. Each strategy gets one full untimed warm-up,
# followed by three independent measured runs by default. Measured strategy order
# alternates by repeat to reduce simple temporal-order bias.

set -euo pipefail

ROOT="${1:?Usage: $0 PREPARED_ROOT [RESULT_ROOT]}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_ROOT="${2:-/tmp/hrhsa-point-bisection-${STAMP}}"
VALIDATION_ROOT="${HRHSA_BISECTION_DATA_ROOT:-${ROOT}/workstation-point-bisection-validation}"
PREPARE_WORKERS="${HRHSA_BISECTION_PREPARE_WORKERS:-4}"
MANAGED_GIB_PER_SHARD="${HRHSA_BISECTION_MANAGED_GIB_PER_SHARD:-16}"
MEASURE_REPEATS="${HRHSA_BISECTION_REPEATS:-3}"

POINTS=100000000
RASTER_GIB=24
GEOMETRY=6x2
SHARDS=6
THREADS_PER_SHARD=2
GRAPH_PARTITIONS=100
SPATIAL_CHUNK=1024
VALIDATION_POINTS=100000

if (( MEASURE_REPEATS <= 0 )); then
    echo "HRHSA_BISECTION_REPEATS must be positive" >&2
    exit 2
fi

mkdir -p "$RESULT_ROOT" "$RESULT_ROOT/warmup"

python benchmarking/scripts/01_python/prepare_point_bisection_workstation.py \
    --root "$ROOT" \
    --validation-root "$VALIDATION_ROOT" \
    --point-count "$POINTS" \
    --raster-gib "$RASTER_GIB" \
    --scenarios uniform,moderate,strong \
    --prepare-workers "$PREPARE_WORKERS"

SUMMARY_REPEATS="$RESULT_ROOT/summary-repeats.jsonl"
SUMMARY="$RESULT_ROOT/summary.jsonl"
rm -f "$SUMMARY_REPEATS" "$SUMMARY"

run_direct() {
    local scenario_root="$1"
    local output="$2"
    rm -f "$output"
    python benchmarking/scripts/01_python/profile_point_partitioned_production.py \
        --root "$scenario_root" \
        --output "$output" \
        --point-count "$POINTS" \
        --raster-gib "$RASTER_GIB" \
        --geometry "$GEOMETRY" \
        --graph-partitions "$GRAPH_PARTITIONS" \
        --spatial-chunk "$SPATIAL_CHUNK" \
        --validation-points "$VALIDATION_POINTS" \
        --repeats 1
}

run_bisection() {
    local scenario_root="$1"
    local output_dir="$2"
    rm -rf "$output_dir"
    python benchmarking/scripts/01_python/launch_point_spatial_balanced_shards.py \
        --root "$scenario_root" \
        --output-dir "$output_dir" \
        --point-count "$POINTS" \
        --raster-gib "$RASTER_GIB" \
        --shard-count "$SHARDS" \
        --workers-per-shard 1 \
        --threads-per-worker "$THREADS_PER_SHARD" \
        --graph-partitions "$GRAPH_PARTITIONS" \
        --spatial-chunk "$SPATIAL_CHUNK" \
        --managed-memory-gib "$MANAGED_GIB_PER_SHARD" \
        --validation-points "$VALIDATION_POINTS" \
        --repeats 1
}

for SCENARIO in uniform moderate strong; do
    SCENARIO_ROOT="$VALIDATION_ROOT/$SCENARIO"
    echo
    echo "================================================================="
    echo "Point bisection validation: $SCENARIO"
    echo "================================================================="

    echo "--- direct full-workload warm-up (not recorded) ---"
    run_direct \
        "$SCENARIO_ROOT" \
        "$RESULT_ROOT/warmup/${SCENARIO}-direct.jsonl"

    echo "--- bisection full-workload warm-up (not recorded) ---"
    run_bisection \
        "$SCENARIO_ROOT" \
        "$RESULT_ROOT/warmup/${SCENARIO}-bisection"

    for REPEAT in $(seq 1 "$MEASURE_REPEATS"); do
        REPEAT_TAG="$(printf '%02d' "$REPEAT")"
        DIRECT_OUT="$RESULT_ROOT/${SCENARIO}-direct-repeat-${REPEAT_TAG}.jsonl"
        BISECTION_OUT="$RESULT_ROOT/${SCENARIO}-bisection-repeat-${REPEAT_TAG}"

        # Alternate D/B and B/D order to reduce simple drift/cache-order bias.
        if (( REPEAT % 2 == 1 )); then
            echo "--- repeat $REPEAT/$MEASURE_REPEATS: direct measured run ---"
            run_direct "$SCENARIO_ROOT" "$DIRECT_OUT"
            echo "--- repeat $REPEAT/$MEASURE_REPEATS: bisection measured run ---"
            run_bisection "$SCENARIO_ROOT" "$BISECTION_OUT"
        else
            echo "--- repeat $REPEAT/$MEASURE_REPEATS: bisection measured run ---"
            run_bisection "$SCENARIO_ROOT" "$BISECTION_OUT"
            echo "--- repeat $REPEAT/$MEASURE_REPEATS: direct measured run ---"
            run_direct "$SCENARIO_ROOT" "$DIRECT_OUT"
        fi

        python benchmarking/scripts/01_python/summarize_point_bisection_workstation.py \
            --scenario-manifest "$SCENARIO_ROOT/bisection_validation_manifest.json" \
            --baseline-jsonl "$DIRECT_OUT" \
            --bisection-dir "$BISECTION_OUT" \
            --output "$SUMMARY_REPEATS"
    done
done

python benchmarking/scripts/01_python/aggregate_point_bisection_repeats.py \
    --input "$SUMMARY_REPEATS" \
    --output "$SUMMARY" \
    --expected-repeats "$MEASURE_REPEATS"

echo
echo "Campaign complete."
echo "Results: $RESULT_ROOT"
echo "Per-repeat A/B records: $SUMMARY_REPEATS"
echo "Median A/B summary: $SUMMARY"
