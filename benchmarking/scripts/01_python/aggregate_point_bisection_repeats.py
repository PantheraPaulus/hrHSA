"""Aggregate repeated workstation point-bisection A/B summaries.

The single-run summarizer emits one direct and one bisection record per paired
measurement.  This helper collapses those records to robust median summaries
while retaining the individual wall times and variability diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No benchmark records in {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _median(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in records if row.get(key) is not None]
    return None if not values else float(np.median(values))


def _copy_constant(records: list[dict[str, Any]], key: str) -> Any:
    values = [row.get(key) for row in records]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise RuntimeError(f"Repeated records disagree on {key}: {values}")
    return first


def _cv(values: list[float]) -> float | None:
    mean = float(np.mean(values))
    return None if mean == 0 else float(np.std(values) / mean)


def _aggregate_strategy(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("Cannot aggregate an empty record group")

    walls = [float(row["sampling_wall_seconds"]) for row in records]
    common_keys = (
        "campaign",
        "scenario",
        "point_count",
        "raster_gib",
        "hot_domain_fraction",
        "hot_point_fraction",
        "partition_rows_cv",
        "partition_rows_max_over_mean",
        "git_commit",
        "strategy",
    )
    result = {key: _copy_constant(records, key) for key in common_keys}
    median_wall = float(np.median(walls))
    result.update(
        {
            "repeats": len(records),
            "sampling_wall_seconds": median_wall,
            "sampling_wall_seconds_repeats": walls,
            "sampling_wall_seconds_min": float(np.min(walls)),
            "sampling_wall_seconds_max": float(np.max(walls)),
            "sampling_wall_seconds_cv": _cv(walls),
            "sampling_plus_planner_seconds": _median(
                records, "sampling_plus_planner_seconds"
            ),
            "throughput_points_s": float(result["point_count"]) / median_wall,
        }
    )

    direct_keys = (
        "peak_process_tree_rss_mb",
        "task_stream_span_seconds",
        "pipeline_minus_task_span_seconds",
        "worker_busy_cores_pipeline",
        "task_compute_parallelism",
        "task_transfer_seconds",
    )
    bisection_keys = (
        "sum_shard_peak_process_tree_rss_mb",
        "planner_seconds_max",
        "planner_seconds_median",
        "shard_rows_min",
        "shard_rows_max",
        "shard_rows_mean",
        "shard_rows_cv",
        "shard_rows_max_over_mean",
        "shard_wall_min_seconds",
        "shard_wall_median_seconds",
        "shard_wall_max_seconds",
        "shard_wall_cv",
        "critical_path_tail_over_median_seconds",
        "worker_busy_cores_sum",
        "task_transfer_seconds_sum",
    )
    keys = direct_keys if result["strategy"] == "direct_partitioned" else bisection_keys
    for key in keys:
        value = _median(records, key)
        if value is not None:
            result[key] = value

    if result["strategy"] != "direct_partitioned":
        result["shard_count"] = _copy_constant(records, "shard_count")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    args = parser.parse_args()

    if args.expected_repeats <= 0:
        parser.error("expected-repeats must be positive")

    rows = _read_jsonl(args.input)
    scenarios = []
    for row in rows:
        scenario = str(row["scenario"])
        if scenario not in scenarios:
            scenarios.append(scenario)

    output: list[dict[str, Any]] = []
    for scenario in scenarios:
        direct = [
            row
            for row in rows
            if row["scenario"] == scenario and row["strategy"] == "direct_partitioned"
        ]
        bisection = [
            row
            for row in rows
            if row["scenario"] == scenario
            and row["strategy"] == "recursive_weighted_spatial_bisection"
        ]
        if len(direct) != args.expected_repeats or len(bisection) != args.expected_repeats:
            raise RuntimeError(
                f"{scenario}: expected {args.expected_repeats} records per strategy; "
                f"found direct={len(direct)}, bisection={len(bisection)}"
            )

        direct_summary = _aggregate_strategy(direct)
        bisection_summary = _aggregate_strategy(bisection)
        direct_wall = float(direct_summary["sampling_wall_seconds"])
        bisection_wall = float(bisection_summary["sampling_wall_seconds"])
        paired_speedups = [
            float(d["sampling_wall_seconds"]) / float(b["sampling_wall_seconds"])
            for d, b in zip(direct, bisection, strict=True)
        ]

        direct_summary["speedup_vs_direct"] = 1.0
        bisection_summary.update(
            {
                "speedup_vs_direct": direct_wall / bisection_wall,
                "wall_reduction_fraction": 1.0 - bisection_wall / direct_wall,
                "paired_speedup_repeats": paired_speedups,
                "paired_speedup_median": float(np.median(paired_speedups)),
                "paired_speedup_min": float(np.min(paired_speedups)),
                "paired_speedup_max": float(np.max(paired_speedups)),
            }
        )
        output.extend([direct_summary, bisection_summary])

    _write_jsonl(args.output, output)

    print(
        f"{'scenario':<10} {'strategy':<40} {'median_s':>10} "
        f"{'min_s':>9} {'max_s':>9} {'M points/s':>12} {'speedup':>9}"
    )
    print("-" * 106)
    for row in output:
        print(
            f"{row['scenario']:<10} {row['strategy']:<40} "
            f"{row['sampling_wall_seconds']:>10.3f} "
            f"{row['sampling_wall_seconds_min']:>9.3f} "
            f"{row['sampling_wall_seconds_max']:>9.3f} "
            f"{row['throughput_points_s'] / 1e6:>12.3f} "
            f"{row['speedup_vs_direct']:>9.3f}"
        )


if __name__ == "__main__":
    main()
