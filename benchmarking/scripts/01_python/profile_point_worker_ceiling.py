"""Measure the worker-side concurrency ceiling for point raster extraction.

The full point pipeline includes Parquet reads, GeoDataFrame construction, point
routing, index-payload scatter and Dask graph submission. This diagnostic removes
most of those stages from the measured interval: it prepares one spatial point
batch once, computes chunk routing once, scatters all packed point-index payloads
once, and then repeatedly measures only submission/execution/gather of the same
raster extraction workload.

If this isolated workload reaches the configured execution-thread count, the full
pipeline is starving Dask workers. If it remains at only a few concurrent tasks,
the remaining ceiling lies in the Dask raster task/data path itself rather than
outer point routing.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import dask
import numpy as np
import psutil

from hsa.compute.raster import (
    _chunk_number,
    _chunk_starts,
    _dimension_chunk_lengths,
    _extract_numpy_block_flat,
    _group_positions_by_chunk,
    _nearest_indices,
)
from hsa.compute.telemetry import (
    aggregate_worker_runtime_delta,
    distributed_worker_runtime_snapshot,
    summarize_task_stream,
)
from hsa.compute.workloads import parse_geometry
from hsa.sampling import _as_raster_dataarray
from run_point_scaling import (
    _iter_prepared_spatial_batches,
    _load_inputs,
    _point_workload_entry,
)
from run_surface_scaling import _local_configuration, _window


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _task_events(client, *, start: float, stop: float):
    try:
        return client.get_task_stream(start=start, stop=stop)
    except Exception:
        return []


def _task_family(key: Any) -> str:
    """Return a compact, stable-enough family label for Dask task keys."""
    if isinstance(key, tuple) and key:
        key = key[0]
    text = str(key)
    if "-" in text:
        return text.split("-", 1)[0]
    return text


def _task_family_summary(events) -> dict[str, dict[str, float | int]]:
    counts: dict[str, int] = defaultdict(int)
    compute: dict[str, float] = defaultdict(float)
    transfer: dict[str, float] = defaultdict(float)

    for event in events or ():
        family = _task_family(event.get("key"))
        counts[family] += 1
        for interval in event.get("startstops", ()) or ():
            action = str(interval.get("action", "unknown"))
            start = interval.get("start")
            stop = interval.get("stop")
            try:
                duration = max(0.0, float(stop) - float(start))
            except (TypeError, ValueError):
                continue
            if action == "compute":
                compute[family] += duration
            elif action == "transfer":
                transfer[family] += duration

    families = sorted(
        counts,
        key=lambda name: (compute.get(name, 0.0), counts[name]),
        reverse=True,
    )
    return {
        family: {
            "tasks": counts[family],
            "compute_seconds": float(compute.get(family, 0.0)),
            "transfer_seconds": float(transfer.get(family, 0.0)),
        }
        for family in families
    }


def _prepare_distributed_specs(points, env, chunks):
    """Prepare the same block/index specifications used by the production sampler."""
    raster = _as_raster_dataarray(env).transpose("band", "y", "x")

    try:
        env_crs = raster.rio.crs
    except Exception:
        env_crs = None
    transformed = points
    if env_crs is not None and points.crs != env_crs:
        transformed = points.to_crs(env_crs)

    x_coordinates = np.asarray(raster["x"].values)
    y_coordinates = np.asarray(raster["y"].values)
    point_x = transformed.geometry.x.to_numpy(dtype=float)
    point_y = transformed.geometry.y.to_numpy(dtype=float)

    row_index = _nearest_indices(y_coordinates, point_y)
    col_index = _nearest_indices(x_coordinates, point_x)

    chunks = dict(chunks)
    chunks["band"] = -1
    if getattr(raster.data, "chunks", None) is not None:
        raster = raster.chunk(chunks)

    y_nominal = int(chunks.get("y", raster.sizes["y"]))
    x_nominal = int(chunks.get("x", raster.sizes["x"]))
    y_lengths = _dimension_chunk_lengths(raster, "y", nominal=y_nominal)
    x_lengths = _dimension_chunk_lengths(raster, "x", nominal=x_nominal)
    y_starts = _chunk_starts(y_lengths)
    x_starts = _chunk_starts(x_lengths)

    chunk_y = _chunk_number(row_index, y_lengths)
    chunk_x = _chunk_number(col_index, x_lengths)
    grouped = _group_positions_by_chunk(
        chunk_y,
        chunk_x,
        n_x_chunks=len(x_lengths),
    )

    specs = []
    for cy, cx, positions in grouped:
        y_start = int(y_starts[cy])
        x_start = int(x_starts[cx])
        y_stop = y_start + int(y_lengths[cy])
        x_stop = x_start + int(x_lengths[cx])
        block_width = int(x_lengths[cx])

        local_rows = row_index[positions] - y_start
        local_cols = col_index[positions] - x_start
        max_flat_index = int(y_lengths[cy]) * block_width - 1
        index_dtype = np.uint32 if max_flat_index <= np.iinfo(np.uint32).max else np.uint64
        flat_indices = (local_rows * block_width + local_cols).astype(
            index_dtype,
            copy=False,
        )
        block = raster.isel(
            y=slice(y_start, y_stop),
            x=slice(x_start, x_stop),
        ).data
        specs.append((block, flat_indices))
    return specs


def _build_tasks(specs, index_futures):
    return [
        dask.delayed(_extract_numpy_block_flat, pure=False)(block, index_future)
        for (block, _), index_future in zip(specs, index_futures)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the isolated worker-side concurrency ceiling for point sampling."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--geometry", default="6x2")
    parser.add_argument("--partitions-per-batch", type=int, default=16)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if min(
        args.point_count,
        args.partitions_per_batch,
        args.spatial_chunk,
        args.chunk_mb,
        args.repeats,
    ) <= 0 or args.warmups < 0:
        parser.error("point/chunk/batch/repeat values must be positive and warmups non-negative")

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
        raise RuntimeError("Worker ceiling profile requires a Dask client.")

    index_futures = []
    try:
        client.wait_for_workers(workers, timeout=args.worker_startup_timeout)
        try:
            client.get_task_stream()
        except Exception:
            pass

        prepared = _iter_prepared_spatial_batches(
            workload_dir=Path(entry["directory"]),
            expected_partitions=int(entry["partitions"]),
            partitions_per_batch=args.partitions_per_batch,
            env=env,
        )
        points, batch_indices, partition_lengths, read_s, assembly_s, geometry_s = next(prepared)
        del prepared

        route_started = perf_counter()
        specs = _prepare_distributed_specs(points, env, chunks)
        routing_seconds = perf_counter() - route_started
        del points
        gc.collect()

        payloads = [indices for _, indices in specs]
        scatter_started = perf_counter()
        index_futures = client.scatter(
            payloads,
            broadcast=False,
            hash=False,
        )
        scatter_seconds = perf_counter() - scatter_started
        del payloads

        print(
            f"prepared one batch: partitions={len(batch_indices)}, "
            f"rows={sum(partition_lengths):,}, chunk_specs={len(specs)}, "
            f"routing={routing_seconds:.3f}s, scatter={scatter_seconds:.3f}s"
        )

        for warmup in range(1, args.warmups + 1):
            tasks = _build_tasks(specs, index_futures)
            futures = client.compute(tasks)
            client.gather(futures)
            client.cancel(futures)
            del tasks, futures
            gc.collect()
            print(f"warm-up {warmup}/{args.warmups} complete")

        for repeat in range(1, args.repeats + 1):
            tasks = _build_tasks(specs, index_futures)
            worker_before = distributed_worker_runtime_snapshot(client)
            task_start = time.time()
            started = perf_counter()
            futures = client.compute(tasks)
            computed = client.gather(futures)
            wall_seconds = perf_counter() - started
            task_stop = time.time()
            worker_after = distributed_worker_runtime_snapshot(client)
            events = _task_events(client, start=task_start, stop=task_stop)

            worker = aggregate_worker_runtime_delta(worker_before, worker_after)
            task_summary = summarize_task_stream(events)
            families = _task_family_summary(events)
            worker_cpu = worker.get("cpu_total_seconds")
            worker_busy = (
                None
                if worker_cpu is None or wall_seconds <= 0
                else float(worker_cpu) / wall_seconds
            )

            record = {
                "repeat": repeat,
                "geometry": args.geometry,
                "workers": workers,
                "threads_per_worker": threads,
                "execution_threads": workers * threads,
                "point_count_family": int(entry["target_points"]),
                "batch_rows": int(sum(partition_lengths)),
                "batch_partitions": len(batch_indices),
                "chunk_specs": len(specs),
                "raster_gib": raster_workload.logical_gib,
                "routing_seconds_precomputed": routing_seconds,
                "scatter_seconds_precomputed": scatter_seconds,
                "isolated_wall_seconds": wall_seconds,
                "worker_cpu_seconds": worker_cpu,
                "worker_busy_cores": worker_busy,
                "worker_execution_thread_utilization_fraction": (
                    None
                    if worker_busy is None
                    else worker_busy / (workers * threads)
                ),
                "task_stream_tasks": task_summary.get("tasks"),
                "task_compute_seconds": task_summary.get("compute_seconds"),
                "task_transfer_seconds": task_summary.get("transfer_seconds"),
                "task_deserialize_seconds": task_summary.get("deserialize_seconds"),
                "task_compute_parallelism": task_summary.get("compute_parallelism"),
                "task_compute_parallelism_fraction_of_execution_threads": (
                    None
                    if task_summary.get("compute_parallelism") is None
                    else task_summary["compute_parallelism"] / (workers * threads)
                ),
                "task_stream_span_seconds": task_summary.get("span_seconds"),
                "task_compute_mean_seconds": task_summary.get("compute_task_mean_seconds"),
                "task_compute_median_seconds": task_summary.get("compute_task_median_seconds"),
                "task_families": families,
            }
            _append_jsonl(output, record)

            print(
                f"repeat {repeat}: isolated={wall_seconds:.3f}s "
                f"worker_cores={worker_busy if worker_busy is not None else 'NA'} "
                f"task_parallelism={task_summary.get('compute_parallelism')}"
            )

            client.cancel(futures)
            del tasks, futures, computed
            gc.collect()
    finally:
        if index_futures:
            client.cancel(index_futures)
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
