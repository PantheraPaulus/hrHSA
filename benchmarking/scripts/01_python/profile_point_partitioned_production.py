"""Benchmark the production Dask-native partitioned point sampler.

The synthetic point family stores normalized ``u``/``v`` coordinates so one
family can be reused across raster-size campaigns.  This runner materializes a
one-time x/y Parquet cache for the selected raster window *before* measurement,
then benchmarks :func:`hsa.compute.iter_sample_raster_stack_partitioned`
directly.  The timed path therefore exercises package code rather than the
mechanism-diagnostic helper implementations.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil

from hsa.compute import (
    PointPartition,
    benchmark_timer,
    iter_sample_raster_stack_partitioned,
    sample_raster_stack_chunked,
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
from profile_point_dask_native_graph import _task_family_summary
from run_point_scaling import _load_inputs, _point_workload_entry
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


def _cache_signature(entry, env) -> dict[str, Any]:
    x = np.asarray(env["x"].values, dtype=np.float64)
    y = np.asarray(env["y"].values, dtype=np.float64)
    return {
        "source_directory": str(Path(entry["directory"]).resolve()),
        "target_points": int(entry["target_points"]),
        "partitions": int(entry["partitions"]),
        "raster_shape": [int(env.sizes["y"]), int(env.sizes["x"])],
        "xmin": float(np.min(x)),
        "xmax": float(np.max(x)),
        "ymin": float(np.min(y)),
        "ymax": float(np.max(y)),
        "crs": str(env.rio.crs),
    }


def _xy_cache_directory(root: Path, entry, env) -> Path:
    return (
        root
        / "points-xy-cache"
        / f"points-{int(entry['target_points'])}"
        / f"raster-{int(env.sizes['y'])}x{int(env.sizes['x'])}"
    )


def _materialize_xy_cache(root: Path, entry, env) -> list[PointPartition]:
    """Create/reuse real-coordinate spatial point partitions outside timing."""
    cache = _xy_cache_directory(root, entry, env)
    manifest_path = cache / "manifest.json"
    signature = _cache_signature(entry, env)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature:
            items = manifest.get("partitions", [])
            if len(items) == int(entry["partitions"]) and all(
                (cache / item["file"]).exists() for item in items
            ):
                print(f"reusing x/y point cache: {cache}")
                return [
                    PointPartition(
                        path=cache / item["file"],
                        bounds=tuple(item["bounds"]),
                        rows=int(item["rows"]),
                        crs=signature["crs"],
                    )
                    for item in items
                ]
        raise RuntimeError(
            f"Existing x/y cache at {cache} does not match this raster window; "
            "remove that cache directory before rebuilding."
        )

    cache.mkdir(parents=True, exist_ok=True)
    source = Path(entry["directory"])
    x = np.asarray(env["x"].values, dtype=np.float64)
    y = np.asarray(env["y"].values, dtype=np.float64)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    items = []
    expected_partitions = int(entry["partitions"])
    for part_index in range(expected_partitions):
        source_path = source / f"part-{part_index:06d}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source point tile: {source_path}")
        frame = pd.read_parquet(
            source_path,
            columns=["point_id", "u", "v", "used"],
        )
        px = xmin + frame["u"].to_numpy(dtype=np.float64) * (xmax - xmin)
        py = ymin + frame["v"].to_numpy(dtype=np.float64) * (ymax - ymin)
        output = pd.DataFrame(
            {
                "x": px,
                "y": py,
                "point_id": frame["point_id"].to_numpy(dtype=np.int64),
                "used": frame["used"].to_numpy(dtype=bool),
            }
        )
        target = cache / f"part-{part_index:06d}.parquet"
        output.to_parquet(target, index=False, compression="zstd")
        if len(output):
            bounds = [
                float(np.min(px)),
                float(np.min(py)),
                float(np.max(px)),
                float(np.max(py)),
            ]
        else:
            bounds = [xmin, ymin, xmin, ymin]
        items.append(
            {
                "file": target.name,
                "rows": int(len(output)),
                "bounds": bounds,
            }
        )
        del frame, output, px, py
        if (part_index + 1) % 10 == 0 or part_index + 1 == expected_partitions:
            print(
                f"prepared x/y point cache {part_index + 1}/{expected_partitions} partitions"
            )

    manifest = {"signature": signature, "partitions": items}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return [
        PointPartition(
            path=cache / item["file"],
            bounds=tuple(item["bounds"]),
            rows=int(item["rows"]),
            crs=signature["crs"],
        )
        for item in items
    ]


def _validate_production_path(
    *,
    partitions: list[PointPartition],
    env,
    client,
    chunks,
    validation_points: int,
) -> None:
    first = partitions[0]
    frame = pd.read_parquet(
        first.path,
        columns=["x", "y", "point_id", "used"],
    ).iloc[:validation_points].copy()
    cache_path = first.path.parent / "validation-production.parquet"
    frame.to_parquet(cache_path, index=False, compression="zstd")
    validation = PointPartition(
        path=cache_path,
        bounds=(
            float(frame["x"].min()),
            float(frame["y"].min()),
            float(frame["x"].max()),
            float(frame["y"].max()),
        ),
        rows=len(frame),
        crs=first.crs,
    )

    points = gpd.GeoDataFrame(
        {
            "point_id": frame["point_id"].to_numpy(dtype=np.int64),
            "used": frame["used"].to_numpy(dtype=bool),
        },
        geometry=gpd.points_from_xy(frame["x"], frame["y"]),
        crs=first.crs,
    )
    reference = sample_raster_stack_chunked(
        points,
        env,
        chunks=chunks,
        client=client,
        id_cols="point_id",
        require_inside=True,
    ).reset_index(drop=True)
    candidate = next(
        iter_sample_raster_stack_partitioned(
            [validation],
            env,
            preserve_cols="used",
            id_cols="point_id",
            chunks=chunks,
            graph_partitions=1,
            client=client,
            require_inside=True,
        )
    ).frame.reset_index(drop=True)
    pd.testing.assert_frame_equal(reference, candidate, check_exact=True)
    print(f"production sampler validation passed for {len(frame):,} points")
    del frame, points, reference, candidate
    gc.collect()


def _run_once(
    *,
    partitions: list[PointPartition],
    env,
    client,
    chunks,
    graph_partitions: int | None,
    workers: int,
    threads_per_worker: int,
) -> dict[str, Any]:
    expected_rows = int(sum(partition.rows or 0 for partition in partitions))
    worker_before = distributed_worker_runtime_snapshot(client)
    driver_before = process_runtime_snapshot()
    node_before = distributed_node_runtime_snapshot(client)
    task_start = time.time()

    rows = 0
    with benchmark_timer(client=client) as timer:
        for sampled in iter_sample_raster_stack_partitioned(
            partitions,
            env,
            preserve_cols="used",
            id_cols="point_id",
            chunks=chunks,
            graph_partitions=graph_partitions,
            client=client,
            require_inside=True,
        ):
            rows += len(sampled.frame)
            del sampled

    task_stop = time.time()
    node_after = distributed_node_runtime_snapshot(client)
    driver_after = process_runtime_snapshot()
    worker_after = distributed_worker_runtime_snapshot(client)
    events = _task_events(client, start=task_start, stop=task_stop)

    if rows != expected_rows:
        raise RuntimeError(
            f"Production partitioned sampler returned {rows:,} rows; "
            f"expected {expected_rows:,}."
        )

    wall = float(timer["wall_seconds"])
    worker = aggregate_worker_runtime_delta(worker_before, worker_after)
    driver = process_runtime_delta(driver_before, driver_after)
    nodes = distributed_node_runtime_delta(node_before, node_after)
    tasks = summarize_task_stream(events)
    families = _task_family_summary(events)
    node = dict(nodes.get("aggregate") or {})

    worker_cpu = worker.get("cpu_total_seconds")
    worker_busy = None if worker_cpu is None or wall <= 0 else float(worker_cpu) / wall
    driver_cpu = driver.get("cpu_total_seconds")
    driver_busy = None if driver_cpu is None or wall <= 0 else float(driver_cpu) / wall
    task_span = tasks.get("span_seconds")
    operation_memory = dict(timer)

    return {
        "pipeline_seconds": wall,
        "throughput_points_s": expected_rows / wall,
        "rows": expected_rows,
        "partitions": len(partitions),
        "graph_partitions": graph_partitions,
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
        "task_stream_span_seconds": task_span,
        "pipeline_minus_task_span_seconds": (
            None if task_span is None else wall - float(task_span)
        ),
        "task_compute_mean_seconds": tasks.get("compute_task_mean_seconds"),
        "task_compute_median_seconds": tasks.get("compute_task_median_seconds"),
        "task_families": families,
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
        description="Benchmark the production partition-native point sampler."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--geometry", default="6x2")
    parser.add_argument("--graph-partitions", type=int, default=100)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if min(
        args.point_count,
        args.graph_partitions,
        args.spatial_chunk,
        args.chunk_mb,
        args.validation_points,
        args.repeats,
    ) <= 0:
        parser.error("point/graph/chunk/validation/repeat values must be positive")

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

    # This is intentionally outside all benchmark timers.
    partitions = _materialize_xy_cache(root, entry, env)

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
        raise RuntimeError("Production partitioned benchmark requires a Dask client.")

    try:
        client.wait_for_workers(workers, timeout=args.worker_startup_timeout)
        try:
            client.get_task_stream()
        except Exception:
            pass

        _validate_production_path(
            partitions=partitions,
            env=env,
            client=client,
            chunks=chunks,
            validation_points=args.validation_points,
        )

        for repeat in range(1, args.repeats + 1):
            record = _run_once(
                partitions=partitions,
                env=env,
                client=client,
                chunks=chunks,
                graph_partitions=args.graph_partitions,
                workers=workers,
                threads_per_worker=threads,
            )
            record.update(
                {
                    "repeat": repeat,
                    "geometry": args.geometry,
                    "workers": workers,
                    "threads_per_worker": threads,
                    "execution_threads": workers * threads,
                    "spatial_chunk": args.spatial_chunk,
                    "raster_gib": raster_workload.logical_gib,
                    "architecture": "production_partitioned_block_local",
                }
            )
            _append_jsonl(output, record)
            print(
                f"repeat {repeat}: "
                f"pipeline={record['pipeline_seconds']:.3f}s "
                f"points/s={record['throughput_points_s']:,.0f} "
                f"worker_cores={record['worker_busy_cores_pipeline']} "
                f"task_parallelism={record['task_compute_parallelism']} "
                f"task_span={record['task_stream_span_seconds']} "
                f"tail={record['pipeline_minus_task_span_seconds']} "
                f"transfer_s={record['task_transfer_seconds']} "
                f"rss_mb={record['operation_peak_process_tree_rss_mb']}"
            )
            gc.collect()
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
