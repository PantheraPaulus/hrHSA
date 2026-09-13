"""Benchmark partition-routed, coalesced Dask submission for point sampling.

This is the second Option-B prototype. Spatial Parquet tiles remain independent
routing/sort units, but several already-routed tiles are coalesced into one Dask
submission graph. This avoids the tiny-graph starvation seen in
``profile_point_partition_streaming.py`` while retaining small routing units and
bounded producer/consumer overlap.

No package behavior is changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import dask
import numpy as np
import pandas as pd
import psutil

from hsa.compute import benchmark_timer
from hsa.compute.raster import _extract_numpy_block_flat
from hsa.compute.scaling_telemetry import (
    distributed_node_runtime_delta,
    distributed_node_runtime_snapshot,
)
from hsa.compute.telemetry import (
    aggregate_worker_runtime_delta,
    distributed_worker_runtime_snapshot,
    process_runtime_delta,
    process_runtime_snapshot,
    summarize_task_stream,
)
from hsa.compute.workloads import parse_geometry
from profile_point_partition_streaming import (
    PreparedPartition,
    RoutingContext,
    SubmittedPartition,
    _append_jsonl,
    _prepare_partition,
    _routing_context,
    _task_events,
    _validate_candidate,
)
from run_point_scaling import _load_inputs, _point_workload_entry
from run_surface_scaling import _local_configuration, _window


def _parse_positive_ints(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        value = int(token.strip())
        if value <= 0:
            raise ValueError("values must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one value is required")
    return values


def _submit_group(
    prepared_group: list[PreparedPartition],
    *,
    client,
) -> tuple[list[SubmittedPartition], float, float]:
    """Submit several independently-routed tiles in one Dask graph."""
    specs = [spec for prepared in prepared_group for spec in prepared.specs]
    payloads = [indices for _, indices in specs]

    scatter_started = perf_counter()
    all_index_futures = list(
        client.scatter(payloads, broadcast=False, hash=False)
    )
    scatter_seconds = perf_counter() - scatter_started

    submit_started = perf_counter()
    tasks = [
        dask.delayed(_extract_numpy_block_flat, pure=False)(block, index_future)
        for (block, _), index_future in zip(specs, all_index_futures)
    ]
    all_extraction_futures = list(client.compute(tasks))
    submit_seconds = perf_counter() - submit_started

    submitted: list[SubmittedPartition] = []
    offset = 0
    for prepared in prepared_group:
        count = len(prepared.specs)
        submitted.append(
            SubmittedPartition(
                prepared=prepared,
                index_futures=all_index_futures[offset : offset + count],
                extraction_futures=all_extraction_futures[offset : offset + count],
            )
        )
        offset += count

    del specs, payloads, tasks
    return submitted, float(scatter_seconds), float(submit_seconds)


def _gather_group(
    submitted_group: list[SubmittedPartition],
    *,
    context: RoutingContext,
    client,
) -> tuple[int, float, float]:
    """Gather one coalesced graph once, then assemble each tile locally."""
    all_futures = [
        future
        for submitted in submitted_group
        for future in submitted.extraction_futures
    ]
    gather_started = perf_counter()
    computed = list(client.gather(all_futures))
    gather_seconds = perf_counter() - gather_started

    output_started = perf_counter()
    rows = 0
    offset = 0
    for submitted in submitted_group:
        prepared = submitted.prepared
        count = len(prepared.positions)
        extracted_group = computed[offset : offset + count]
        offset += count

        values = np.empty(
            (len(context.sampled_bands), prepared.rows),
            dtype=context.dtype,
        )
        for positions, extracted in zip(prepared.positions, extracted_group):
            values[:, positions] = np.asarray(extracted, dtype=context.dtype)

        # Materialize the same local result shape/columns as production. We do
        # not retain it because this benchmark matches write_root=None.
        out = pd.DataFrame(
            values.T,
            columns=list(context.sampled_bands),
            index=prepared.index,
        )
        out["x"] = prepared.point_x
        out["y"] = prepared.point_y
        out["used"] = prepared.used
        out["point_id"] = prepared.point_id
        rows += len(out)
        del values, out, extracted_group

    output_seconds = perf_counter() - output_started
    del computed
    return rows, float(gather_seconds), float(output_seconds)


def _release_group(submitted_group: list[SubmittedPartition], *, client) -> None:
    futures = [
        future
        for submitted in submitted_group
        for future in (*submitted.index_futures, *submitted.extraction_futures)
    ]
    if futures:
        client.cancel(futures)


def _run_once(
    *,
    entry,
    context: RoutingContext,
    client,
    routing_threads: int,
    prepare_ahead: int,
    submission_group_partitions: int,
    groups_in_flight: int,
    workers: int,
    threads_per_worker: int,
) -> dict[str, Any]:
    workload_dir = Path(entry["directory"])
    expected_partitions = int(entry["partitions"])
    expected_rows = int(entry["target_points"])

    read_seconds_sum = 0.0
    geometry_seconds_sum = 0.0
    routing_seconds_sum = 0.0
    scatter_seconds_sum = 0.0
    submit_seconds_sum = 0.0
    gather_seconds_sum = 0.0
    output_seconds_sum = 0.0
    chunk_specs = 0
    rows_prepared = 0
    rows_assembled = 0
    peak_pending_groups = 0
    peak_pending_partitions = 0

    worker_before = distributed_worker_runtime_snapshot(client)
    driver_before = process_runtime_snapshot()
    node_before = distributed_node_runtime_snapshot(client)
    task_start = time.time()

    preparation: dict[int, Future] = {}
    next_to_schedule = 0
    pending_groups: deque[list[SubmittedPartition]] = deque()

    def schedule_one(executor: ThreadPoolExecutor, part_index: int) -> None:
        preparation[part_index] = executor.submit(
            _prepare_partition,
            workload_dir=workload_dir,
            part_index=part_index,
            context=context,
        )

    def refill(executor: ThreadPoolExecutor) -> None:
        nonlocal next_to_schedule
        while (
            next_to_schedule < expected_partitions
            and len(preparation) < prepare_ahead
        ):
            schedule_one(executor, next_to_schedule)
            next_to_schedule += 1

    with benchmark_timer(client=client) as timer:
        with ThreadPoolExecutor(
            max_workers=routing_threads,
            thread_name_prefix="hrhsa-point-route",
        ) as executor:
            refill(executor)
            part_index = 0
            while part_index < expected_partitions:
                prepared_group: list[PreparedPartition] = []
                group_stop = min(
                    part_index + submission_group_partitions,
                    expected_partitions,
                )
                while part_index < group_stop:
                    prepared = preparation.pop(part_index).result()
                    refill(executor)
                    read_seconds_sum += prepared.read_seconds
                    geometry_seconds_sum += prepared.geometry_seconds
                    routing_seconds_sum += prepared.routing_seconds
                    rows_prepared += prepared.rows
                    chunk_specs += len(prepared.specs)
                    prepared_group.append(prepared)
                    part_index += 1

                submitted_group, scatter_s, submit_s = _submit_group(
                    prepared_group,
                    client=client,
                )
                scatter_seconds_sum += scatter_s
                submit_seconds_sum += submit_s
                pending_groups.append(submitted_group)
                peak_pending_groups = max(peak_pending_groups, len(pending_groups))
                peak_pending_partitions = max(
                    peak_pending_partitions,
                    sum(len(group) for group in pending_groups),
                )

                if len(pending_groups) >= groups_in_flight:
                    oldest = pending_groups.popleft()
                    try:
                        assembled, gather_s, output_s = _gather_group(
                            oldest,
                            context=context,
                            client=client,
                        )
                        rows_assembled += assembled
                        gather_seconds_sum += gather_s
                        output_seconds_sum += output_s
                    finally:
                        _release_group(oldest, client=client)
                    del oldest

            while pending_groups:
                group = pending_groups.popleft()
                try:
                    assembled, gather_s, output_s = _gather_group(
                        group,
                        context=context,
                        client=client,
                    )
                    rows_assembled += assembled
                    gather_seconds_sum += gather_s
                    output_seconds_sum += output_s
                finally:
                    _release_group(group, client=client)
                del group

    task_stop = time.time()
    node_after = distributed_node_runtime_snapshot(client)
    driver_after = process_runtime_snapshot()
    worker_after = distributed_worker_runtime_snapshot(client)
    events = _task_events(client, start=task_start, stop=task_stop)

    if rows_prepared != expected_rows or rows_assembled != expected_rows:
        raise RuntimeError(
            "Partition-streamed row count changed: "
            f"prepared={rows_prepared:,}, assembled={rows_assembled:,}, "
            f"expected={expected_rows:,}."
        )

    wall = float(timer["wall_seconds"])
    worker = aggregate_worker_runtime_delta(worker_before, worker_after)
    driver = process_runtime_delta(driver_before, driver_after)
    nodes = distributed_node_runtime_delta(node_before, node_after)
    tasks = summarize_task_stream(events)
    node = dict(nodes.get("aggregate") or {})

    worker_cpu = worker.get("cpu_total_seconds")
    worker_busy = (
        None if worker_cpu is None or wall <= 0 else float(worker_cpu) / wall
    )
    driver_cpu = driver.get("cpu_total_seconds")
    driver_busy = (
        None if driver_cpu is None or wall <= 0 else float(driver_cpu) / wall
    )

    operation_memory = dict(timer)
    return {
        "pipeline_seconds": wall,
        "throughput_points_s": expected_rows / wall,
        "rows": expected_rows,
        "partitions": expected_partitions,
        "routing_threads": int(routing_threads),
        "prepare_ahead": int(prepare_ahead),
        "submission_group_partitions": int(submission_group_partitions),
        "groups_in_flight": int(groups_in_flight),
        "peak_pending_groups": int(peak_pending_groups),
        "peak_pending_partitions": int(peak_pending_partitions),
        "chunk_specs_total": int(chunk_specs),
        "chunk_specs_mean_per_partition": chunk_specs / expected_partitions,
        "read_seconds_sum": float(read_seconds_sum),
        "geometry_seconds_sum": float(geometry_seconds_sum),
        "routing_seconds_sum": float(routing_seconds_sum),
        "scatter_seconds_sum": float(scatter_seconds_sum),
        "submit_seconds_sum": float(submit_seconds_sum),
        "gather_seconds_sum": float(gather_seconds_sum),
        "output_seconds_sum": float(output_seconds_sum),
        "worker_cpu_seconds": worker_cpu,
        "worker_busy_cores_pipeline": worker_busy,
        "worker_execution_thread_utilization_fraction": (
            None if worker_busy is None else worker_busy / (workers * threads_per_worker)
        ),
        "driver_cpu_seconds": driver_cpu,
        "driver_busy_cores_pipeline": driver_busy,
        "task_stream_tasks": tasks.get("tasks"),
        "task_compute_seconds": tasks.get("compute_seconds"),
        "task_transfer_seconds": tasks.get("transfer_seconds"),
        "task_deserialize_seconds": tasks.get("deserialize_seconds"),
        "task_compute_parallelism": tasks.get("compute_parallelism"),
        "task_stream_span_seconds": tasks.get("span_seconds"),
        "task_compute_mean_seconds": tasks.get("compute_task_mean_seconds"),
        "task_compute_median_seconds": tasks.get("compute_task_median_seconds"),
        "node_cpu_busy_fraction_mean": node.get("mean_cpu_busy_fraction"),
        "node_cpu_iowait_fraction_mean": node.get("mean_cpu_iowait_fraction"),
        "node_disk_read_bytes": node.get("disk_read_bytes"),
        "node_major_page_faults": node.get("major_page_faults"),
        "operation_peak_process_tree_rss_mb": operation_memory.get(
            "operation_peak_process_tree_rss_mb"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark independently-routed tiles with coalesced Dask submission."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--geometry", default="6x2")
    parser.add_argument("--routing-threads", type=int, default=4)
    parser.add_argument(
        "--submission-group-partitions",
        default="2,4,8,16",
        help="Comma-separated number of independently-routed tiles per Dask graph.",
    )
    parser.add_argument("--groups-in-flight", type=int, default=2)
    parser.add_argument("--prepare-ahead", type=int, default=32)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    group_sizes = _parse_positive_ints(args.submission_group_partitions)
    if min(
        args.point_count,
        args.routing_threads,
        args.groups_in_flight,
        args.prepare_ahead,
        args.spatial_chunk,
        args.chunk_mb,
        args.validation_points,
        args.repeats,
    ) <= 0:
        parser.error("all point/routing/group/chunk/repeat values must be positive")

    workers, threads = parse_geometry(args.geometry)
    physical = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    if workers * threads > physical:
        parser.error(
            f"Geometry {args.geometry} needs {workers * threads} physical cores; "
            f"machine reports {physical}."
        )

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    full_env, surface_manifest, point_manifest = _load_inputs(root)
    entry = _point_workload_entry(point_manifest, args.point_count)
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", args.spatial_chunk
        )
    )
    env, raster_workload = _window(full_env, args.raster_gib, storage_chunk)
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}
    context = _routing_context(env, chunks)

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
        raise RuntimeError("Coalesced partition profile requires a Dask client.")

    try:
        client.wait_for_workers(workers, timeout=args.worker_startup_timeout)
        try:
            client.get_task_stream()
        except Exception:
            pass

        _validate_candidate(
            entry=entry,
            context=context,
            client=client,
            chunks=chunks,
            validation_points=args.validation_points,
        )
        print(f"validation passed for {args.validation_points:,} points")

        for repeat in range(1, args.repeats + 1):
            ordered = group_sizes if repeat % 2 else list(reversed(group_sizes))
            for group_size in ordered:
                result = _run_once(
                    entry=entry,
                    context=context,
                    client=client,
                    routing_threads=args.routing_threads,
                    prepare_ahead=max(args.prepare_ahead, group_size),
                    submission_group_partitions=group_size,
                    groups_in_flight=args.groups_in_flight,
                    workers=workers,
                    threads_per_worker=threads,
                )
                record = {
                    **result,
                    "repeat": repeat,
                    "geometry": args.geometry,
                    "workers": workers,
                    "threads_per_worker": threads,
                    "execution_threads": workers * threads,
                    "spatial_chunk": args.spatial_chunk,
                    "raster_gib": raster_workload.logical_gib,
                }
                _append_jsonl(output, record)
                print(
                    f"repeat {repeat} group={group_size}: "
                    f"pipeline={record['pipeline_seconds']:.3f}s "
                    f"points/s={record['throughput_points_s']:,.0f} "
                    f"worker_cores={record['worker_busy_cores_pipeline']} "
                    f"task_parallelism={record['task_compute_parallelism']} "
                    f"driver_cores={record['driver_busy_cores_pipeline']}"
                )
                gc.collect()
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
