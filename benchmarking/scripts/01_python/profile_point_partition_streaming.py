"""Benchmark partition-streamed routing/submission for point raster sampling.

The production point-scaling runner concatenates several spatial Parquet tiles into
one large GeoDataFrame and submits one blocking sampling call per outer batch.
This experimental companion keeps the prepared ~1M-row spatial tiles independent:

1. a bounded producer pool reads, geometrizes and routes tiles independently;
2. each completed tile is immediately scattered/submitted to Dask;
3. previously submitted extraction tasks execute while later tiles are prepared;
4. only a bounded number of sampled tiles remain outstanding;
5. results are assembled per tile and discarded, matching the no-write benchmark.

No package behavior is changed. The script first validates the experimental path
against ``sample_raster_stack_chunked`` on a subset of the first prepared tile.
The main purpose is to determine whether tile-level streaming can turn the worker
ceiling demonstrated by ``profile_point_worker_ceiling.py`` into end-to-end
parallelism without unbounded memory growth.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil

from hsa.compute import benchmark_timer, sample_raster_stack_chunked
from hsa.compute.raster import (
    _chunk_number,
    _chunk_starts,
    _dimension_chunk_lengths,
    _extract_numpy_block_flat,
    _group_positions_by_chunk,
    _nearest_indices,
)
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
from hsa.sampling import _as_raster_dataarray
from run_point_scaling import (
    _load_inputs,
    _point_frame_to_gdf,
    _point_workload_entry,
)
from run_surface_scaling import _local_configuration, _window


@dataclass(frozen=True)
class RoutingContext:
    raster: Any
    env: Any
    env_crs: Any
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    y_lengths: tuple[int, ...]
    x_lengths: tuple[int, ...]
    y_starts: np.ndarray
    x_starts: np.ndarray
    sampled_bands: tuple[str, ...]
    dtype: np.dtype


@dataclass
class PreparedPartition:
    part_index: int
    rows: int
    index: pd.Index
    point_id: np.ndarray
    used: np.ndarray
    point_x: np.ndarray
    point_y: np.ndarray
    positions: list[np.ndarray]
    specs: list[tuple[Any, np.ndarray]]
    read_seconds: float
    geometry_seconds: float
    routing_seconds: float


@dataclass
class SubmittedPartition:
    prepared: PreparedPartition
    index_futures: list[Any]
    extraction_futures: list[Any]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _task_events(client, *, start: float, stop: float):
    try:
        return client.get_task_stream(start=start, stop=stop)
    except Exception:
        return []


def _routing_context(env, chunks, *, dtype: str | np.dtype = "float32") -> RoutingContext:
    """Build immutable raster-routing metadata once for the complete campaign."""
    raster = _as_raster_dataarray(env).transpose("band", "y", "x")
    try:
        env_crs = raster.rio.crs
    except Exception:
        env_crs = None

    chunks = dict(chunks)
    chunks["band"] = -1
    if getattr(raster.data, "chunks", None) is not None:
        raster = raster.chunk(chunks)

    y_nominal = int(chunks.get("y", raster.sizes["y"]))
    x_nominal = int(chunks.get("x", raster.sizes["x"]))
    y_lengths = _dimension_chunk_lengths(raster, "y", nominal=y_nominal)
    x_lengths = _dimension_chunk_lengths(raster, "x", nominal=x_nominal)

    return RoutingContext(
        raster=raster,
        env=env,
        env_crs=env_crs,
        x_coordinates=np.asarray(raster["x"].values),
        y_coordinates=np.asarray(raster["y"].values),
        y_lengths=y_lengths,
        x_lengths=x_lengths,
        y_starts=_chunk_starts(y_lengths),
        x_starts=_chunk_starts(x_lengths),
        sampled_bands=tuple(str(value) for value in raster["band"].values),
        dtype=np.dtype(dtype),
    )


def _route_points(points: gpd.GeoDataFrame, context: RoutingContext):
    """Route one already-geometrized spatial tile to raster chunk specifications."""
    transformed = points
    if context.env_crs is not None and points.crs != context.env_crs:
        transformed = points.to_crs(context.env_crs)

    point_x = transformed.geometry.x.to_numpy(dtype=float)
    point_y = transformed.geometry.y.to_numpy(dtype=float)
    row_index = _nearest_indices(context.y_coordinates, point_y)
    col_index = _nearest_indices(context.x_coordinates, point_x)

    chunk_y = _chunk_number(row_index, context.y_lengths)
    chunk_x = _chunk_number(col_index, context.x_lengths)
    grouped = _group_positions_by_chunk(
        chunk_y,
        chunk_x,
        n_x_chunks=len(context.x_lengths),
    )

    specs: list[tuple[Any, np.ndarray]] = []
    positions_out: list[np.ndarray] = []
    for cy, cx, positions in grouped:
        y_start = int(context.y_starts[cy])
        x_start = int(context.x_starts[cx])
        y_stop = y_start + int(context.y_lengths[cy])
        x_stop = x_start + int(context.x_lengths[cx])
        block_width = int(context.x_lengths[cx])

        local_rows = row_index[positions] - y_start
        local_cols = col_index[positions] - x_start
        max_flat_index = int(context.y_lengths[cy]) * block_width - 1
        index_dtype = (
            np.uint32
            if max_flat_index <= np.iinfo(np.uint32).max
            else np.uint64
        )
        flat_indices = (
            local_rows * block_width + local_cols
        ).astype(index_dtype, copy=False)
        block = context.raster.isel(
            y=slice(y_start, y_stop),
            x=slice(x_start, x_stop),
        ).data
        specs.append((block, flat_indices))
        positions_out.append(positions)

    return point_x, point_y, positions_out, specs


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    part_index: int,
    context: RoutingContext,
    read_seconds: float = 0.0,
) -> PreparedPartition:
    geometry_started = perf_counter()
    points = _point_frame_to_gdf(frame, context.env)
    geometry_seconds = perf_counter() - geometry_started

    routing_started = perf_counter()
    point_x, point_y, positions, specs = _route_points(points, context)
    routing_seconds = perf_counter() - routing_started

    prepared = PreparedPartition(
        part_index=int(part_index),
        rows=len(points),
        index=points.index.copy(),
        point_id=points["point_id"].to_numpy(dtype=np.int64, copy=True),
        used=points["used"].to_numpy(dtype=bool, copy=True),
        point_x=point_x,
        point_y=point_y,
        positions=positions,
        specs=specs,
        read_seconds=float(read_seconds),
        geometry_seconds=float(geometry_seconds),
        routing_seconds=float(routing_seconds),
    )
    del points
    return prepared


def _prepare_partition(
    *,
    workload_dir: Path,
    part_index: int,
    context: RoutingContext,
) -> PreparedPartition:
    path = workload_dir / f"part-{part_index:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing point tile: {path}")

    started = perf_counter()
    frame = pd.read_parquet(
        path,
        columns=["point_id", "u", "v", "used"],
    )
    read_seconds = perf_counter() - started
    try:
        return _prepare_frame(
            frame,
            part_index=part_index,
            context=context,
            read_seconds=read_seconds,
        )
    finally:
        del frame


def _submit_partition(prepared: PreparedPartition, *, client) -> tuple[SubmittedPartition, float, float]:
    """Scatter one tile's compact index payloads and immediately submit extraction."""
    payloads = [indices for _, indices in prepared.specs]
    scatter_started = perf_counter()
    index_futures = client.scatter(
        payloads,
        broadcast=False,
        hash=False,
    )
    scatter_seconds = perf_counter() - scatter_started

    submit_started = perf_counter()
    tasks = [
        dask.delayed(_extract_numpy_block_flat, pure=False)(block, index_future)
        for (block, _), index_future in zip(prepared.specs, index_futures)
    ]
    extraction_futures = client.compute(tasks)
    submit_seconds = perf_counter() - submit_started
    del payloads, tasks

    return (
        SubmittedPartition(
            prepared=prepared,
            index_futures=list(index_futures),
            extraction_futures=list(extraction_futures),
        ),
        float(scatter_seconds),
        float(submit_seconds),
    )


def _assemble_partition(
    submitted: SubmittedPartition,
    *,
    context: RoutingContext,
    client,
) -> tuple[pd.DataFrame, float, float]:
    """Gather and assemble one tile in the exact production output column order."""
    gather_started = perf_counter()
    computed = client.gather(submitted.extraction_futures)
    gather_seconds = perf_counter() - gather_started

    output_started = perf_counter()
    prepared = submitted.prepared
    values = np.empty(
        (len(context.sampled_bands), prepared.rows),
        dtype=context.dtype,
    )
    for positions, extracted in zip(prepared.positions, computed):
        values[:, positions] = np.asarray(extracted, dtype=context.dtype)

    out = pd.DataFrame(
        values.T,
        columns=list(context.sampled_bands),
        index=prepared.index,
    )
    out["x"] = prepared.point_x
    out["y"] = prepared.point_y
    out["used"] = prepared.used
    out["point_id"] = prepared.point_id
    output_seconds = perf_counter() - output_started
    del values, computed
    return out, float(gather_seconds), float(output_seconds)


def _release_submitted(submitted: SubmittedPartition, *, client) -> None:
    futures = [*submitted.index_futures, *submitted.extraction_futures]
    if futures:
        client.cancel(futures)


def _validate_candidate(
    *,
    entry,
    context: RoutingContext,
    client,
    chunks,
    validation_points: int,
) -> None:
    path = Path(entry["directory"]) / "part-000000.parquet"
    frame = pd.read_parquet(
        path,
        columns=["point_id", "u", "v", "used"],
    )
    if len(frame) > validation_points:
        frame = frame.iloc[:validation_points].copy()

    points = _point_frame_to_gdf(frame, context.env)
    reference = sample_raster_stack_chunked(
        points,
        context.env,
        chunks=chunks,
        client=client,
        id_cols="point_id",
        require_inside=True,
    )

    prepared = _prepare_frame(
        frame,
        part_index=0,
        context=context,
    )
    submitted, _, _ = _submit_partition(prepared, client=client)
    try:
        candidate, _, _ = _assemble_partition(
            submitted,
            context=context,
            client=client,
        )
        pd.testing.assert_frame_equal(reference, candidate, check_exact=True)
    finally:
        _release_submitted(submitted, client=client)

    del frame, points, reference, candidate, prepared, submitted
    gc.collect()


def _run_streamed(
    *,
    entry,
    context: RoutingContext,
    client,
    routing_threads: int,
    prepare_ahead: int,
    partitions_in_flight: int,
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
    rows = 0
    peak_pending_partitions = 0

    worker_before = distributed_worker_runtime_snapshot(client)
    driver_before = process_runtime_snapshot()
    node_before = distributed_node_runtime_snapshot(client)
    task_start = time.time()

    pending: deque[SubmittedPartition] = deque()
    preparation: dict[int, Future] = {}
    next_to_schedule = 0

    def schedule_one(executor: ThreadPoolExecutor, part_index: int) -> None:
        preparation[part_index] = executor.submit(
            _prepare_partition,
            workload_dir=workload_dir,
            part_index=part_index,
            context=context,
        )

    with benchmark_timer(client=client) as timer:
        with ThreadPoolExecutor(
            max_workers=routing_threads,
            thread_name_prefix="hrhsa-point-route",
        ) as executor:
            while (
                next_to_schedule < expected_partitions
                and len(preparation) < prepare_ahead
            ):
                schedule_one(executor, next_to_schedule)
                next_to_schedule += 1

            for part_index in range(expected_partitions):
                prepared = preparation.pop(part_index).result()

                while (
                    next_to_schedule < expected_partitions
                    and len(preparation) < prepare_ahead
                ):
                    schedule_one(executor, next_to_schedule)
                    next_to_schedule += 1

                read_seconds_sum += prepared.read_seconds
                geometry_seconds_sum += prepared.geometry_seconds
                routing_seconds_sum += prepared.routing_seconds
                rows += prepared.rows
                chunk_specs += len(prepared.specs)

                submitted, scatter_s, submit_s = _submit_partition(
                    prepared,
                    client=client,
                )
                scatter_seconds_sum += scatter_s
                submit_seconds_sum += submit_s
                pending.append(submitted)
                peak_pending_partitions = max(
                    peak_pending_partitions,
                    len(pending),
                )

                if len(pending) >= partitions_in_flight:
                    oldest = pending.popleft()
                    try:
                        sampled, gather_s, output_s = _assemble_partition(
                            oldest,
                            context=context,
                            client=client,
                        )
                        gather_seconds_sum += gather_s
                        output_seconds_sum += output_s
                        if len(sampled) != oldest.prepared.rows:
                            raise RuntimeError("Sampled tile row count changed during assembly.")
                    finally:
                        _release_submitted(oldest, client=client)
                    del sampled, oldest

            while pending:
                submitted = pending.popleft()
                try:
                    sampled, gather_s, output_s = _assemble_partition(
                        submitted,
                        context=context,
                        client=client,
                    )
                    gather_seconds_sum += gather_s
                    output_seconds_sum += output_s
                    if len(sampled) != submitted.prepared.rows:
                        raise RuntimeError("Sampled tile row count changed during assembly.")
                finally:
                    _release_submitted(submitted, client=client)
                del sampled, submitted

    task_stop = time.time()
    node_after = distributed_node_runtime_snapshot(client)
    driver_after = process_runtime_snapshot()
    worker_after = distributed_worker_runtime_snapshot(client)
    events = _task_events(client, start=task_start, stop=task_stop)

    if rows != expected_rows:
        raise RuntimeError(
            f"Streamed workload produced {rows:,} rows; expected {expected_rows:,}."
        )

    pipeline_seconds = float(timer["wall_seconds"])
    worker = aggregate_worker_runtime_delta(worker_before, worker_after)
    driver = process_runtime_delta(driver_before, driver_after)
    nodes = distributed_node_runtime_delta(node_before, node_after)
    tasks = summarize_task_stream(events)
    aggregate = dict(nodes.get("aggregate") or {})

    worker_cpu = worker.get("cpu_total_seconds")
    worker_busy = (
        None
        if worker_cpu is None or pipeline_seconds <= 0
        else float(worker_cpu) / pipeline_seconds
    )
    driver_cpu = driver.get("cpu_total_seconds")
    driver_busy = (
        None
        if driver_cpu is None or pipeline_seconds <= 0
        else float(driver_cpu) / pipeline_seconds
    )
    task_parallelism = tasks.get("compute_parallelism")
    execution_threads = workers * threads_per_worker

    return {
        "pipeline_seconds": pipeline_seconds,
        "throughput_points_s": expected_rows / pipeline_seconds,
        "point_count": expected_rows,
        "partitions": expected_partitions,
        "routing_threads": int(routing_threads),
        "prepare_ahead": int(prepare_ahead),
        "partitions_in_flight": int(partitions_in_flight),
        "peak_pending_partitions": int(peak_pending_partitions),
        "chunk_specs_total": int(chunk_specs),
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
            None if worker_busy is None else worker_busy / execution_threads
        ),
        "driver_cpu_seconds": driver_cpu,
        "driver_busy_cores_pipeline": driver_busy,
        "task_stream_tasks": tasks.get("tasks"),
        "task_compute_seconds": tasks.get("compute_seconds"),
        "task_transfer_seconds": tasks.get("transfer_seconds"),
        "task_deserialize_seconds": tasks.get("deserialize_seconds"),
        "task_compute_parallelism": task_parallelism,
        "task_compute_parallelism_fraction_of_execution_threads": (
            None
            if task_parallelism is None
            else float(task_parallelism) / execution_threads
        ),
        "task_stream_span_seconds": tasks.get("span_seconds"),
        "task_compute_mean_seconds": tasks.get("compute_task_mean_seconds"),
        "task_compute_median_seconds": tasks.get("compute_task_median_seconds"),
        "task_stream_nbytes_observed": tasks.get("nbytes_observed"),
        "node_cpu_busy_fraction_mean": aggregate.get("mean_cpu_busy_fraction"),
        "node_cpu_iowait_fraction_mean": aggregate.get("mean_cpu_iowait_fraction"),
        "node_disk_read_bytes": aggregate.get("disk_read_bytes"),
        "node_major_page_faults": aggregate.get("major_page_faults"),
        "node_workingset_refaults": aggregate.get("workingset_refaults"),
        "operation_peak_process_tree_rss_mb": timer.get(
            "operation_peak_process_tree_rss_mb"
        ),
    }


def _parse_positive_values(text: str, *, name: str) -> list[int]:
    result: list[int] = []
    for token in text.split(","):
        value = int(token.strip())
        if value <= 0:
            raise ValueError(f"{name} values must be positive")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError(f"at least one {name} value is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark independent spatial-tile routing with bounded Dask submission."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--geometry", default="6x2")
    parser.add_argument(
        "--routing-threads",
        default="1,2,4",
        help="Comma-separated producer thread counts for tile read/geometry/routing.",
    )
    parser.add_argument(
        "--partitions-in-flight",
        type=int,
        default=8,
        help="Maximum submitted tiles retained before the oldest is gathered.",
    )
    parser.add_argument(
        "--prepare-ahead",
        type=int,
        default=8,
        help="Maximum tile-preparation futures retained at once.",
    )
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    routing_threads_values = _parse_positive_values(
        args.routing_threads,
        name="routing-threads",
    )
    if min(
        args.point_count,
        args.partitions_in_flight,
        args.prepare_ahead,
        args.spatial_chunk,
        args.chunk_mb,
        args.validation_points,
        args.repeats,
    ) <= 0:
        parser.error("point/chunk/queue/validation/repeat values must be positive")
    if args.prepare_ahead < max(routing_threads_values):
        parser.error("--prepare-ahead must be at least the largest routing thread count")

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
        raise RuntimeError("Partition-streaming profile requires a Dask client.")

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
            ordered = (
                routing_threads_values
                if repeat % 2
                else list(reversed(routing_threads_values))
            )
            for routing_threads in ordered:
                gc.collect()
                result = _run_streamed(
                    entry=entry,
                    context=context,
                    client=client,
                    routing_threads=routing_threads,
                    prepare_ahead=args.prepare_ahead,
                    partitions_in_flight=args.partitions_in_flight,
                    workers=workers,
                    threads_per_worker=threads,
                )
                record = {
                    **result,
                    "repeat": repeat,
                    "strategy": "partition_streaming",
                    "geometry": args.geometry,
                    "workers": workers,
                    "threads_per_worker": threads,
                    "execution_threads": workers * threads,
                    "spatial_chunk": args.spatial_chunk,
                    "raster_gib": raster_workload.logical_gib,
                }
                _append_jsonl(output, record)
                print(
                    f"repeat {repeat} routing_threads={routing_threads}: "
                    f"pipeline={record['pipeline_seconds']:.3f}s "
                    f"points/s={record['throughput_points_s']:,.0f} "
                    f"worker_cores={record['worker_busy_cores_pipeline']} "
                    f"task_parallelism={record['task_compute_parallelism']} "
                    f"driver_cores={record['driver_busy_cores_pipeline']}"
                )
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
