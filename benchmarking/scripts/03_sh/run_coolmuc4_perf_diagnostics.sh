#!/bin/bash
# Best-effort PMU diagnostics for the resident-block direct NumPy kernel.
# Usage: run_coolmuc4_perf_diagnostics.sh DIRECT_ROOT OUTPUT_DIR
set -u

ROOT="${1:?DIRECT_ROOT required}"
OUT="${2:?OUTPUT_DIR required}"
mkdir -p "$OUT"

if ! command -v perf >/dev/null 2>&1; then
    echo "perf unavailable" > "$OUT/perf-unavailable.txt"
    exit 0
fi

perf list > "$OUT/perf-list.txt" 2>&1 || true

for threads in 1 14 56 112; do
    echo "perf detail: ${threads} threads"
    srun --exact --nodes=1 --ntasks=1 --cpus-per-task=112 --cpu-bind=cores \
      perf stat -d -d -r 3 -o "$OUT/perf-detail-${threads}t.txt" -- \
      python benchmarking/scripts/01_python/diagnose_numpy_pinned_scaling.py \
        --root "$ROOT" \
        --output "$OUT/perf-detail-${threads}t.jsonl" \
        --threads "$threads" \
        --block-size 1024 \
        --resident-blocks 16 \
        --tasks 2048 \
        --repeats 1 \
        --warmup-tasks 128 \
        --compute-dtype float32
    echo $? > "$OUT/perf-detail-${threads}t.exitcode"

    echo "perf topdown: ${threads} threads"
    srun --exact --nodes=1 --ntasks=1 --cpus-per-task=112 --cpu-bind=cores \
      perf stat --topdown -r 3 -o "$OUT/perf-topdown-${threads}t.txt" -- \
      python benchmarking/scripts/01_python/diagnose_numpy_pinned_scaling.py \
        --root "$ROOT" \
        --output "$OUT/perf-topdown-${threads}t.jsonl" \
        --threads "$threads" \
        --block-size 1024 \
        --resident-blocks 16 \
        --tasks 2048 \
        --repeats 1 \
        --warmup-tasks 128 \
        --compute-dtype float32
    echo $? > "$OUT/perf-topdown-${threads}t.exitcode"
done

exit 0
