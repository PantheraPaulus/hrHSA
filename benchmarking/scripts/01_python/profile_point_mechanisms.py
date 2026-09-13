"""Diagnose the remaining parallelism limits in large point sampling.

This is a mechanism-oriented companion to ``profile_point_batch_concurrency.py``.
It executes the exact production point-scaling workload path while collecting
worker, driver, scheduler, node and Dask task-stream telemetry. The purpose is to
distinguish task starvation/orchestration gaps from scheduler pressure,
serialization/data movement, and genuinely saturated worker execution.

The compact output JSONL is convenient for comparing repeats. Full nested
telemetry is written beside it as ``<output>.diagnostics.jsonl``.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from hsa.compute.scaling_telemetry import (
    distributed_node_runtime_delta,
    distributed_node_runtime_snapshot,
    scheduler_process_delta,
    scheduler_process_snapshot,
)
from hsa.compute.telemetry import (
    aggregate_worker_runtime_delta,
    distributed_worker_runtime_snapshot,
    process_runtime_delta,
    process_runtime_snapshot,
    summarize_task_stream,
)
from hsa.compute.workloads import parse_geometry
from run_point_scaling import (
    _load_inputs,
    _point_workload_entry,
    _process_point_workload,
    _sample_frame,
)
from run_surface_scaling import _local_configuration, _window


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _scheduler_snapshot(client) -> dict[str, Any]:
    try:
        return dict(client.run_on_scheduler(scheduler_process_snapshot))
    except Exception as exc:
        return {"available": False, "reason": f"scheduler snapshot failed: {exc}"}


def _task_events(client, *, start: float, stop: float):
    try:
        return client.get_task_stream(start=start, stop=stop)
    except Exception:
        return []


def _run_once(
    *,
    entry,
    env,
    client,
    chunks,
    partitions_per_batch: int,
    batches_in_flight: int,
    workers: int,
    threads_per_worker: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    worker_before = distributed_worker_runtime_snapshot(client)
    driver_before = process_runtime_snapshot()
    scheduler_before = _scheduler_snapshot(client)
    node_before = distributed_node_runtime_snapshot(client)
    task_start = time.time()

    result = _process_point_workload(
        entry=entry,
        env=env,
        client=client,
        chunks=chunks,
        write_root=None,
        run_key="point-mechanism-profile",
        partitions_per_batch=partitions_per_batch,
        batches_in_flight=batches_in_flight,
    )
    task_stop = time.time()

    # Snapshot while the operation has just completed, before cleanup or another
    # repeat changes process state. All snapshot overhead lies outside pipeline_seconds.
    node_after = distributed_node_runtime_snapshot(client)
    scheduler_after = _scheduler_snapshot(client)
    driver_after = process_runtime_snapshot()
    worker_after = distributed_worker_runtime_snapshot(client)
    events = _task_events(client, start=task_start, stop=task_stop)

    pipeline_seconds = float(result["pipeline_seconds"])
    worker = aggregate_worker_runtime_delta(worker_before, worker_after)
    driver = process_runtime_delta(driver_before, driver_after)
    scheduler = scheduler_process_delta(
        scheduler_before,
        scheduler_after,
        wall_seconds=pipeline_seconds,
    )
    nodes = distributed_node_runtime_delta(node_before, node_after)
    tasks = summarize_task_stream(events)
    aggregate = dict(nodes.get("aggregate") or {})

    worker_cpu_seconds = worker.get("cpu_total_seconds")
    worker_busy_cores = (
        None
        if worker_cpu_seconds is None or pipeline_seconds <= 0
        else float(worker_cpu_seconds) / pipeline_seconds
    )
    execution_threads = int(workers * threads_per_worker)
    worker_thread_utilization = (
        None
        if worker_busy_cores is None or execution_threads <= 0
        else float(worker_busy_cores) / execution_threads
    )

    driver_cpu_seconds = driver.get("cpu_total_seconds")
    driver_busy_cores = (
        None
        if driver_cpu_seconds is None or pipeline_seconds <= 0
        else float(driver_cpu_seconds) / pipeline_seconds
    )

    scheduler_cpu_seconds = scheduler.get("cpu_total_seconds")
    scheduler_busy_core = (
        None
        if scheduler_cpu_seconds is None or pipeline_seconds <= 0
        else float(scheduler_cpu_seconds) / pipeline_seconds
    )

    task_parallelism = tasks.get("compute_parallelism")
    task_parallelism_fraction = (
        None
        if task_parallelism is None or execution_threads <= 0
        else float(task_parallelism) / execution_threads
    )

    operation_memory = dict(result.get("operation_memory") or {})
    expected_rows = int(entry["target_points"])
    expected_partitions = int(entry["partitions"])

    summary = {
        "point_count": expected_rows,
        "partitions": expected_partitions,
        "partitions_per_batch": int(result["partitions_per_batch"]),
        "batches": int(result["batches"]),
        "batches_in_flight": int(result["batches_in_flight"]),
        "pipeline_seconds": pipeline_seconds,
        "throughput_points_s": expected_rows / pipeline_seconds,
        "read_seconds_sum": float(result["read_seconds"]),
        "assembly_seconds_sum": float(result["assembly_seconds"]),
        "geometry_seconds_sum": float(result["geometry_seconds"]),
        "sampling_call_seconds_sum": float(result["sampling_seconds"]),
        "sampling_call_seconds_max": result["sampling_call_seconds_max"],
        "sampling_overlap_window_seconds": result[
            "sampling_overlap_window_seconds"
        ],
        "sampling_overlap_factor": result["sampling_overlap_factor"],
        "worker_cpu_seconds": worker_cpu_seconds,
        "worker_busy_cores_pipeline": worker_busy_cores,
        "worker_execution_thread_utilization_fraction": worker_thread_utilization,
        "worker_context_switches": worker.get("context_switches"),
        "worker_involuntary_context_switches": worker.get(
            "involuntary_context_switches"
        ),
        "driver_cpu_seconds": driver_cpu_seconds,
        "driver_busy_cores_pipeline": driver_busy_cores,
        "driver_context_switches": driver.get("context_switches"),
        "scheduler_cpu_seconds": scheduler_cpu_seconds,
        "scheduler_cpu_fraction_of_one_core": scheduler_busy_core,
        "scheduler_context_switches": scheduler.get("context_switches"),
        "task_stream_tasks": tasks.get("tasks"),
        "task_compute_seconds": tasks.get("compute_seconds"),
        "task_transfer_seconds": tasks.get("transfer_seconds"),
        "task_deserialize_seconds": tasks.get("deserialize_seconds"),
        "task_compute_parallelism": task_parallelism,
        "task_compute_parallelism_fraction_of_execution_threads": (
            task_parallelism_fraction
        ),
        "task_compute_mean_seconds": tasks.get("compute_task_mean_seconds"),
        "task_compute_median_seconds": tasks.get("compute_task_median_seconds"),
        "task_compute_min_seconds": tasks.get("compute_task_min_seconds"),
        "task_compute_max_seconds": tasks.get("compute_task_max_seconds"),
        "task_stream_span_seconds": tasks.get("span_seconds"),
        "task_stream_nbytes_observed": tasks.get("nbytes_observed"),
        "node_cpu_busy_fraction_mean": aggregate.get("mean_cpu_busy_fraction"),
        "node_cpu_iowait_fraction_mean": aggregate.get("mean_cpu_iowait_fraction"),
        "node_disk_read_bytes": aggregate.get("disk_read_bytes"),
        "node_disk_write_bytes": aggregate.get("disk_write_bytes"),
        "node_context_switches": aggregate.get("ctx_switches"),
        "node_major_page_faults": aggregate.get("major_page_faults"),
        "node_page_faults": aggregate.get("page_faults"),
        "node_workingset_refaults": aggregate.get("workingset_refaults"),
        "operation_peak_process_tree_rss_mb": operation_memory.get(
            "operation_peak_process_tree_rss_mb"
        ),
    }

    diagnostic = {
        "pipeline": dict(result),
        "worker": worker,
        "driver": driver,
        "scheduler": scheduler,
        "nodes": nodes,
        "task_stream": tasks,
    }
    return summary, diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose point-sampling worker/scheduler/driver parallelism."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--geometry", default="6x2")
    parser.add_argument("--partitions-per-batch", type=int, default=16)
    parser.add_argument("--batches-in-flight", type=int, default=4)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--warmup-points", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    if min(
        args.point_count,
        args.partitions_per_batch,
        args.batches_in_flight,
        args.spatial_chunk,
        args.chunk_mb,
        args.warmup_points,
        args.repeats,
    ) <= 0:
        parser.error("point/chunk/batch/concurrency/repeat values must be positive")

    workers, threads = parse_geometry(args.geometry)
    physical = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    if workers * threads > physical:
        parser.error(
            f"Geometry {args.geometry} needs {workers * threads} physical cores; "
            f"machine reports {physical}."
        )

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    diagnostics_output = output.with_suffix(output.suffix + ".diagnostics.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    full_env, surface_manifest, point_manifest = _load_inputs(root)
    entry = _point_workload_entry(point_manifest, args.point_count)
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", args.spatial_chunk
        )
    )
    env, raster_workload = _window(full_env, args.raster_gib, storage_chunk)
    chunks = {
        "band": -1,
        "y": args.spatial_chunk,
        "x": args.spatial_chunk,
    }

    total_memory_gib = psutil.virtual_memory().total / 1024**3
    local_tmp = root / "dask-tmp-points"
    local_tmp.mkdir(parents=True, exist_ok=True)
    execution = _local_configuration(
        workers=workers,
        threads=threads,
        total_memory_gib=total_memory_gib,
        managed_memory_fraction=args.managed_memory_fraction,
        local_directory=local_tmp,
        chunk_mb=args.chunk_mb,
        worker_startup_timeout=args.worker_startup_timeout,
    )
    client, cluster = execution.create_client()
    if client is None:
        raise RuntimeError("Point mechanism profile requires a Dask client.")

    try:
        client.wait_for_workers(workers, timeout=args.worker_startup_timeout)

        # Initialize Dask's task-stream plugin before either warm-up or measurement.
        try:
            client.get_task_stream()
        except Exception:
            pass

        warmup_path = Path(entry["directory"]) / "part-000000.parquet"
        warmup_frame = pd.read_parquet(
            warmup_path,
            columns=["point_id", "u", "v", "used"],
        )
        if len(warmup_frame) > args.warmup_points:
            warmup_frame = warmup_frame.iloc[: args.warmup_points].copy()
        warmup_points, warmup_sampled = _sample_frame(
            warmup_frame,
            env=env,
            client=client,
            chunks=chunks,
        )
        del warmup_frame, warmup_points, warmup_sampled
        gc.collect()

        for repeat in range(1, args.repeats + 1):
            summary, diagnostic = _run_once(
                entry=entry,
                env=env,
                client=client,
                chunks=chunks,
                partitions_per_batch=args.partitions_per_batch,
                batches_in_flight=args.batches_in_flight,
                workers=workers,
                threads_per_worker=threads,
            )
            record = {
                **summary,
                "repeat": repeat,
                "geometry": args.geometry,
                "workers": workers,
                "threads_per_worker": threads,
                "execution_threads": workers * threads,
                "spatial_chunk": args.spatial_chunk,
                "raster_gib": raster_workload.logical_gib,
                "diagnostics_jsonl": str(diagnostics_output),
            }
            diagnostic_record = {
                "repeat": repeat,
                "geometry": args.geometry,
                "workers": workers,
                "threads_per_worker": threads,
                "execution_threads": workers * threads,
                "point_count": int(entry["target_points"]),
                "partitions_per_batch": args.partitions_per_batch,
                "batches_in_flight": args.batches_in_flight,
                "spatial_chunk": args.spatial_chunk,
                "raster_gib": raster_workload.logical_gib,
                **diagnostic,
            }
            _append_jsonl(output, record)
            _append_jsonl(diagnostics_output, diagnostic_record)

            task_parallelism = record.get("task_compute_parallelism")
            driver_busy = record.get("driver_busy_cores_pipeline")
            scheduler_busy = record.get("scheduler_cpu_fraction_of_one_core")
            print(
                f"repeat {repeat}: pipeline={record['pipeline_seconds']:.3f}s "
                f"points/s={record['throughput_points_s']:,.0f} "
                f"worker_cores={record['worker_busy_cores_pipeline']:.2f} "
                f"task_parallelism={task_parallelism if task_parallelism is not None else 'NA'} "
                f"driver_cores={driver_busy if driver_busy is not None else 'NA'} "
                f"scheduler_core={scheduler_busy if scheduler_busy is not None else 'NA'}"
            )
            gc.collect()
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
