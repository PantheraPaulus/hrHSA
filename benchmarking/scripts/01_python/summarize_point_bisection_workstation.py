"""Summarize direct-vs-bisection workstation point-sampling measurements.

The bisection launcher starts all shard timers behind a filesystem barrier, so the
critical-path wall for one measured repeat is the maximum shard pipeline time,
not the sum.  Planner/setup cost is kept separate and also reported in a
conservative ``sampling_plus_planner_seconds`` metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No benchmark records in {path}")
    return rows


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _safe_cv(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    return None if mean == 0 else float(np.std(array) / mean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--bisection-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenario = json.loads(args.scenario_manifest.read_text(encoding="utf-8"))
    baseline_records = _read_jsonl(args.baseline_jsonl)
    baseline = baseline_records[-1]

    shard_paths = sorted(args.bisection_dir.glob("shard-*.jsonl"))
    if not shard_paths:
        raise RuntimeError(f"No shard JSONL files under {args.bisection_dir}")
    shard_records = [_read_jsonl(path)[-1] for path in shard_paths]
    setup_paths = sorted(args.bisection_dir.glob("shard-*.setup.json"))
    setup = [json.loads(path.read_text(encoding="utf-8")) for path in setup_paths]

    expected_shards = int(shard_records[0]["shard_count"])
    if len(shard_records) != expected_shards:
        raise RuntimeError(
            f"Expected {expected_shards} shard records, found {len(shard_records)}"
        )
    if len({int(record["repeat"]) for record in shard_records}) != 1:
        raise RuntimeError("Shard records do not describe the same repeat")

    total_rows = int(sum(int(record["rows"]) for record in shard_records))
    expected_rows = int(scenario["target_points"])
    if total_rows != expected_rows:
        raise RuntimeError(
            f"Bisection shards returned {total_rows:,} rows; expected {expected_rows:,}"
        )
    if int(baseline["rows"]) != expected_rows:
        raise RuntimeError("Direct baseline row count differs from the scenario workload")

    shard_walls = [float(record["pipeline_seconds"]) for record in shard_records]
    shard_rows = [int(record["rows"]) for record in shard_records]
    bisection_wall = max(shard_walls)
    planner_seconds = [float(item.get("shard_plan_seconds", 0.0)) for item in setup]
    planner_max = max(planner_seconds) if planner_seconds else None
    planner_median = float(np.median(planner_seconds)) if planner_seconds else None

    shard_peak_rss = [
        float(record["operation_peak_process_tree_rss_mb"])
        for record in shard_records
        if record.get("operation_peak_process_tree_rss_mb") is not None
    ]
    bisection_peak_rss_sum = float(sum(shard_peak_rss)) if shard_peak_rss else None
    baseline_wall = float(baseline["pipeline_seconds"])
    baseline_peak_rss = baseline.get("operation_peak_process_tree_rss_mb")
    speedup = baseline_wall / bisection_wall

    common = {
        "campaign": "workstation-point-bisection-validation-v1",
        "scenario": scenario["scenario"],
        "point_count": expected_rows,
        "raster_gib": float(scenario["raster_gib"]),
        "hot_domain_fraction": scenario.get("hot_domain_fraction"),
        "hot_point_fraction": scenario.get("hot_point_fraction"),
        "partition_rows_cv": float(scenario["partition_rows_cv"]),
        "partition_rows_max_over_mean": float(scenario["partition_rows_max_over_mean"]),
        "git_commit": _git_commit(),
    }

    direct_record = {
        **common,
        "strategy": "direct_partitioned",
        "sampling_wall_seconds": baseline_wall,
        "sampling_plus_planner_seconds": baseline_wall,
        "throughput_points_s": expected_rows / baseline_wall,
        "peak_process_tree_rss_mb": baseline_peak_rss,
        "task_stream_span_seconds": baseline.get("task_stream_span_seconds"),
        "pipeline_minus_task_span_seconds": baseline.get(
            "pipeline_minus_task_span_seconds"
        ),
        "worker_busy_cores_pipeline": baseline.get("worker_busy_cores_pipeline"),
        "task_compute_parallelism": baseline.get("task_compute_parallelism"),
        "task_transfer_seconds": baseline.get("task_transfer_seconds"),
        "speedup_vs_direct": 1.0,
    }
    bisection_record = {
        **common,
        "strategy": "recursive_weighted_spatial_bisection",
        "shard_count": expected_shards,
        "sampling_wall_seconds": bisection_wall,
        "sampling_plus_planner_seconds": (
            None if planner_max is None else bisection_wall + planner_max
        ),
        "throughput_points_s": expected_rows / bisection_wall,
        "sum_shard_peak_process_tree_rss_mb": bisection_peak_rss_sum,
        "planner_seconds_max": planner_max,
        "planner_seconds_median": planner_median,
        "shard_rows_min": min(shard_rows),
        "shard_rows_max": max(shard_rows),
        "shard_rows_mean": float(np.mean(shard_rows)),
        "shard_rows_cv": _safe_cv([float(value) for value in shard_rows]),
        "shard_rows_max_over_mean": float(max(shard_rows) / np.mean(shard_rows)),
        "shard_wall_min_seconds": min(shard_walls),
        "shard_wall_median_seconds": float(np.median(shard_walls)),
        "shard_wall_max_seconds": max(shard_walls),
        "shard_wall_cv": _safe_cv(shard_walls),
        "critical_path_tail_over_median_seconds": float(
            max(shard_walls) - np.median(shard_walls)
        ),
        "worker_busy_cores_sum": float(
            sum(
                float(record.get("worker_busy_cores_pipeline") or 0.0)
                for record in shard_records
            )
        ),
        "task_transfer_seconds_sum": float(
            sum(float(record.get("task_transfer_seconds") or 0.0) for record in shard_records)
        ),
        "speedup_vs_direct": speedup,
        "wall_reduction_fraction": 1.0 - (bisection_wall / baseline_wall),
    }

    _append_jsonl(args.output, direct_record)
    _append_jsonl(args.output, bisection_record)

    print(
        f"{scenario['scenario']}: direct={baseline_wall:.3f}s "
        f"bisection={bisection_wall:.3f}s speedup={speedup:.3f}x "
        f"shard_rows_cv={bisection_record['shard_rows_cv']:.4f} "
        f"tail={bisection_record['critical_path_tail_over_median_seconds']:.3f}s"
    )
    if planner_max is not None:
        print(f"planner max={planner_max:.6f}s (reported outside sampling wall)")


if __name__ == "__main__":
    main()
