"""Aggregate concurrent spatial point-shard benchmark records.

The two shard pipelines run concurrently on one node.  Effective sampling wall
for one repeat is therefore the maximum shard pipeline time, while total rows
and worker CPU seconds add across shards.  The summary intentionally labels
combined task parallelism as approximate because the independent Dask task
streams do not share an exact common timestamp origin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(args.shard_dir.glob("shard-*.jsonl"))
    if len(files) < 2:
        raise RuntimeError(
            f"Expected at least two shard result files under {args.shard_dir}; found {files}."
        )

    records = []
    for path in files:
        records.extend(_read_jsonl(path))
    repeats = sorted({int(record["repeat"]) for record in records})
    shard_count = max(int(record["shard_count"]) for record in records)

    baseline_by_repeat = {}
    if args.baseline is not None and args.baseline.exists():
        for record in _read_jsonl(args.baseline):
            baseline_by_repeat[int(record["repeat"])] = record

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)

    for repeat in repeats:
        group = [record for record in records if int(record["repeat"]) == repeat]
        if len(group) != shard_count:
            raise RuntimeError(
                f"Repeat {repeat} has {len(group)} shard records; expected {shard_count}."
            )
        indices = sorted(int(record["shard_index"]) for record in group)
        if indices != list(range(shard_count)):
            raise RuntimeError(
                f"Repeat {repeat} shard indices are {indices}; expected 0..{shard_count - 1}."
            )

        wall = max(float(record["pipeline_seconds"]) for record in group)
        rows = sum(int(record["rows"]) for record in group)
        worker_cpu_values = [record.get("worker_cpu_seconds") for record in group]
        worker_cpu = (
            None
            if any(value is None for value in worker_cpu_values)
            else sum(float(value) for value in worker_cpu_values)
        )
        task_compute_values = [record.get("task_compute_seconds") for record in group]
        task_compute = (
            None
            if any(value is None for value in task_compute_values)
            else sum(float(value) for value in task_compute_values)
        )
        task_transfer_values = [record.get("task_transfer_seconds") for record in group]
        task_transfer = (
            None
            if any(value is None for value in task_transfer_values)
            else sum(float(value) for value in task_transfer_values)
        )
        task_spans = [record.get("task_stream_span_seconds") for record in group]
        task_span = (
            None
            if any(value is None for value in task_spans)
            else max(float(value) for value in task_spans)
        )
        process_tree_peaks = [
            record.get("operation_peak_process_tree_rss_mb") for record in group
        ]
        rss_sum = (
            None
            if any(value is None for value in process_tree_peaks)
            else sum(float(value) for value in process_tree_peaks)
        )
        execution_threads = sum(int(record["execution_threads"]) for record in group)

        summary = {
            "repeat": repeat,
            "architecture": "production_partitioned_block_local_spatial_shards",
            "shard_count": shard_count,
            "pipeline_seconds": wall,
            "throughput_points_s": rows / wall,
            "rows": rows,
            "partitions": sum(int(record["partitions"]) for record in group),
            "graph_partitions_per_shard": sorted(
                {int(record["graph_partitions"]) for record in group}
            ),
            "workers": sum(int(record["workers"]) for record in group),
            "execution_threads": execution_threads,
            "worker_cpu_seconds": worker_cpu,
            "worker_busy_cores_pipeline": (
                None if worker_cpu is None else worker_cpu / wall
            ),
            "worker_execution_thread_utilization_fraction": (
                None if worker_cpu is None else worker_cpu / wall / execution_threads
            ),
            "task_compute_seconds": task_compute,
            "task_transfer_seconds": task_transfer,
            "task_stream_span_seconds_approx": task_span,
            "task_compute_parallelism_approx": (
                None
                if task_compute is None or task_span is None or task_span <= 0
                else task_compute / task_span
            ),
            "operation_peak_process_tree_rss_mb_sum": rss_sum,
            "raster_gib_full": max(float(record["raster_gib_full"]) for record in group),
            "raster_gib_shards_sum": sum(
                float(record["raster_gib_shard"]) for record in group
            ),
            "shard_pipeline_seconds": [
                float(record["pipeline_seconds"])
                for record in sorted(group, key=lambda item: int(item["shard_index"]))
            ],
            "shard_throughput_points_s": [
                float(record["throughput_points_s"])
                for record in sorted(group, key=lambda item: int(item["shard_index"]))
            ],
        }

        baseline = baseline_by_repeat.get(repeat)
        if baseline is not None:
            baseline_wall = float(baseline["pipeline_seconds"])
            summary.update(
                {
                    "baseline_pipeline_seconds": baseline_wall,
                    "baseline_throughput_points_s": float(
                        baseline["throughput_points_s"]
                    ),
                    "speedup_vs_monolithic": baseline_wall / wall,
                    "wall_reduction_fraction_vs_monolithic": 1.0 - wall / baseline_wall,
                }
            )

        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, default=str) + "\n")

        print(
            f"repeat {repeat}: sharded={wall:.3f}s "
            f"throughput={rows / wall:,.0f} points/s "
            f"busy_cores={summary['worker_busy_cores_pipeline']}"
        )
        if baseline is not None:
            print(
                f"  monolithic={baseline_wall:.3f}s "
                f"speedup={summary['speedup_vs_monolithic']:.3f}x "
                f"wall_reduction={100 * summary['wall_reduction_fraction_vs_monolithic']:.1f}%"
            )


if __name__ == "__main__":
    main()
