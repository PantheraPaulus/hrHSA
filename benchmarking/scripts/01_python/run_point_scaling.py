"""Run spatially tiled point capacity, strong and weak scaling benchmarks.

Each exact point-count workload is stored in its own spatially tiled Parquet
directory. The runner processes bounded batches of spatial tiles, so neither
the complete input nor output is gathered on the driver. Combining several
nearby spatial partitions in one sampling call exposes enough independent
raster chunks to utilize larger Dask execution budgets while preserving
bounded memory use and spatial locality.

Several bounded sampling calls may optionally be kept in flight concurrently.
This overlaps driver-side preparation/gather latency with useful worker execution
without allowing the complete workload to accumulate in memory.

Two records are emitted per measured repeat:

* ``point_<mode>_sampling_kernel`` — summed elapsed sampling-call time across
  bounded point batches (which may overlap when concurrency is greater than one);
* ``point_<mode>_pipeline`` — complete read -> batch assembly -> geometry ->
  sample -> optional partitioned Parquet write time.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import rioxarray  # noqa: F401 - registers xarray .rio accessor
import xarray as xr

from hsa.compute import (
    COOLMUC4_POLICY,
    append_benchmark_record,
    benchmark_timer,
    coolmuc4_plan,
    current_slurm_allocation,
    discover_runtime_topology,
    iter_sample_raster_stack_chunked,
    make_benchmark_record,
    sample_raster_stack_chunked,
    slurm_allocation_client,
    spatial_task_count,
    validate_worker_topology,
)
from hsa.compute.scaling_telemetry import (
    distributed_node_runtime_delta,
    distributed_node_runtime_snapshot,
)
from hsa.compute.telemetry import (
    aggregate_worker_runtime_delta,
    distributed_worker_runtime_snapshot,
)
from hsa.compute.workloads import (
    geometry_for_core_budget,
    parse_geometry,
    parse_positive_ints,
)
from run_planned_execution import _allocation_physical_cores_per_node
from run_surface_scaling import _local_configuration, _window


def _load_inputs(root: Path):
    surface_manifest_path = root / "surface_family_manifest.json"
    point_manifest_path = root / "point_family_manifest.json"
    if not surface_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {surface_manifest_path}; prepare the raster family first."
        )
    if not point_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {point_manifest_path}; run prepare_point_family.py first."
        )
    surface_manifest = json.loads(surface_manifest_path.read_text(encoding="utf-8"))
    point_manifest = json.loads(point_manifest_path.read_text(encoding="utf-8"))
    ds = xr.open_zarr(root / "environment.zarr", chunks={}, decode_coords="all")
    env = ds["environment"]
    if env.rio.crs is None:
        env = env.rio.write_crs("EPSG:3857")
    return env, surface_manifest, point_manifest


def _point_workload_entry(
    point_manifest: dict[str, Any],
    target_points: int,
) -> dict[str, Any]:
    for item in point_manifest.get("workloads", []):
        if int(item.get("target_points", -1)) == int(target_points):
            return dict(item)
    raise RuntimeError(
        f"No prepared spatial point workload for {int(target_points):,} points. "
        "Add that exact target with prepare_point_family.py."
    )


def _point_frame_to_gdf(frame: pd.DataFrame, env: xr.DataArray) -> gpd.GeoDataFrame:
    x = np.asarray(env["x"].values, dtype=np.float64)
    y = np.asarray(env["y"].values, dtype=np.float64)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    px = xmin + frame["u"].to_numpy(dtype=np.float64) * (xmax - xmin)
    py = ymin + frame["v"].to_numpy(dtype=np.float64) * (ymax - ymin)

    crs = env.rio.crs
    if crs is None:
        raise RuntimeError("Benchmark raster has no CRS.")
    return gpd.GeoDataFrame(
        {
            "point_id": frame["point_id"].to_numpy(dtype=np.int64),
            "used": frame["used"].to_numpy(dtype=bool),
        },
        geometry=gpd.points_from_xy(px, py),
        crs=crs,
    )


def _sample_frame(frame, *, env, client, chunks):
    points = _point_frame_to_gdf(frame, env)
    sampled = next(
        iter_sample_raster_stack_chunked(
            points,
            env,
            batch_size=len(points),
            chunks=chunks,
            client=client,
            id_cols="point_id",
            require_inside=True,
        )
    )
    return points, sampled


def _warmup_first_tile(
    *,
    entry: dict[str, Any],
    env,
    client,
    chunks,
    max_points: int,
) -> int:
    path = Path(entry["directory"]) / "part-000000.parquet"
    frame = pd.read_parquet(path, columns=["point_id", "u", "v", "used"])
    if len(frame) > max_points:
        frame = frame.iloc[:max_points].copy()
    points, sampled = _sample_frame(frame, env=env, client=client, chunks=chunks)
    rows = len(frame)
    del frame, points, sampled
    gc.collect()
    return rows


def _telemetry_summary(
    *,
    worker_before,
    worker_after,
    node_before,
    node_after,
    wall_seconds: float,
    workers: int,
    threads_per_worker: int,
) -> dict[str, Any]:
    worker = aggregate_worker_runtime_delta(worker_before, worker_after)
    nodes = distributed_node_runtime_delta(node_before, node_after)
    worker_cpu = worker.get("cpu_total_seconds")
    busy = None
    utilization = None
    if worker_cpu is not None and wall_seconds > 0:
        busy = worker_cpu / wall_seconds
        utilization = busy / (workers * threads_per_worker)

    aggregate = dict(nodes.get("aggregate") or {})
    return {
        "worker_cpu_seconds": worker_cpu,
        "worker_busy_execution_threads": busy,
        "worker_execution_thread_utilization_fraction": utilization,
        "worker_context_switches": worker.get("context_switches"),
        "worker_involuntary_context_switches": worker.get(
            "involuntary_context_switches"
        ),
        "node_count_observed": len(nodes.get("hosts_matched") or []),
        "node_cpu_busy_fraction_mean": aggregate.get("mean_cpu_busy_fraction"),
        "node_cpu_iowait_fraction_mean": aggregate.get("mean_cpu_iowait_fraction"),
        "node_disk_read_bytes": aggregate.get("disk_read_bytes"),
        "node_disk_write_bytes": aggregate.get("disk_write_bytes"),
        "node_major_page_faults": aggregate.get("major_page_faults"),
        "node_page_scans": aggregate.get("page_scans"),
        "node_workingset_refaults": aggregate.get("workingset_refaults"),
        "node_swap_pages_in": aggregate.get("swap_pages_in"),
        "node_swap_pages_out": aggregate.get("swap_pages_out"),
    }


def _iter_prepared_spatial_batches(
    *,
    workload_dir: Path,
    expected_partitions: int,
    partitions_per_batch: int,
    env,
) -> Iterator[tuple[gpd.GeoDataFrame, list[int], list[int], float, float, float]]:
    """Read, assemble and geometrize one bounded spatial batch at a time."""
    for batch_start in range(0, expected_partitions, partitions_per_batch):
        batch_stop = min(batch_start + partitions_per_batch, expected_partitions)
        batch_indices = list(range(batch_start, batch_stop))
        frames = []
        partition_lengths = []
        read_seconds = 0.0

        for part_index in batch_indices:
            path = workload_dir / f"part-{part_index:06d}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing point tile: {path}")
            started = perf_counter()
            part = pd.read_parquet(
                path,
                columns=["point_id", "u", "v", "used"],
            )
            read_seconds += perf_counter() - started
            partition_lengths.append(len(part))
            frames.append(part)

        assembly_seconds = 0.0
        if len(frames) == 1:
            frame = frames[0]
        else:
            started = perf_counter()
            frame = pd.concat(frames, ignore_index=True)
            assembly_seconds = perf_counter() - started

        started = perf_counter()
        points = _point_frame_to_gdf(frame, env)
        geometry_seconds = perf_counter() - started
        del frames, frame

        yield (
            points,
            batch_indices,
            partition_lengths,
            read_seconds,
            assembly_seconds,
            geometry_seconds,
        )


def _sample_points(*, points, env, client, chunks):
    """Run one blocking production sampling call with operation-local timing."""
    started = perf_counter()
    sampled = sample_raster_stack_chunked(
        points,
        env,
        chunks=chunks,
        client=client,
        id_cols="point_id",
        require_inside=True,
    )
    return perf_counter() - started, sampled


def _write_sampled_batch(
    *,
    sampled: pd.DataFrame,
    target_dir: Path | None,
    batch_indices: list[int],
    partition_lengths: list[int],
) -> float:
    if target_dir is None:
        return 0.0

    started = perf_counter()
    offset = 0
    for part_index, part_rows in zip(batch_indices, partition_lengths):
        sampled.iloc[offset : offset + part_rows].to_parquet(
            target_dir / f"part-{part_index:06d}.parquet",
            index=False,
            compression="zstd",
        )
        offset += part_rows
    return perf_counter() - started


def _process_point_workload(
    *,
    entry: dict[str, Any],
    env,
    client,
    chunks,
    write_root: Path | None,
    run_key: str,
    partitions_per_batch: int,
    batches_in_flight: int = 1,
):
    if batches_in_flight <= 0:
        raise ValueError("batches_in_flight must be positive")

    read_seconds = 0.0
    assembly_seconds = 0.0
    geometry_seconds = 0.0
    sampling_call_seconds: list[float] = []
    write_seconds = 0.0
    rows = 0
    partitions = 0
    batches = 0

    target_dir = None
    if write_root is not None:
        target_dir = write_root / run_key
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    workload_dir = Path(entry["directory"])
    expected_partitions = int(entry["partitions"])
    expected_rows = int(entry["target_points"])
    partitions_per_batch = min(int(partitions_per_batch), expected_partitions)

    sampling_window_started = None
    sampling_window_stopped = None

    with benchmark_timer(client=client) as timer:
        prepared = _iter_prepared_spatial_batches(
            workload_dir=workload_dir,
            expected_partitions=expected_partitions,
            partitions_per_batch=partitions_per_batch,
            env=env,
        )

        if batches_in_flight == 1:
            for (
                points,
                batch_indices,
                partition_lengths,
                read_s,
                assembly_s,
                geometry_s,
            ) in prepared:
                read_seconds += read_s
                assembly_seconds += assembly_s
                geometry_seconds += geometry_s
                rows += len(points)
                partitions += len(batch_indices)
                batches += 1

                if sampling_window_started is None:
                    sampling_window_started = perf_counter()
                sample_s, sampled = _sample_points(
                    points=points,
                    env=env,
                    client=client,
                    chunks=chunks,
                )
                sampling_call_seconds.append(float(sample_s))
                write_seconds += _write_sampled_batch(
                    sampled=sampled,
                    target_dir=target_dir,
                    batch_indices=batch_indices,
                    partition_lengths=partition_lengths,
                )
                del points, sampled
            sampling_window_stopped = perf_counter()
        else:
            executor = ThreadPoolExecutor(
                max_workers=int(batches_in_flight),
                thread_name_prefix="hrhsa-point-batch",
            )
            pending: deque[
                tuple[Future, list[int], list[int]]
            ] = deque()
            try:
                for (
                    points,
                    batch_indices,
                    partition_lengths,
                    read_s,
                    assembly_s,
                    geometry_s,
                ) in prepared:
                    read_seconds += read_s
                    assembly_seconds += assembly_s
                    geometry_seconds += geometry_s
                    rows += len(points)
                    partitions += len(batch_indices)
                    batches += 1

                    if sampling_window_started is None:
                        sampling_window_started = perf_counter()
                    future = executor.submit(
                        _sample_points,
                        points=points,
                        env=env,
                        client=client,
                        chunks=chunks,
                    )
                    pending.append((future, batch_indices, partition_lengths))
                    del points

                    if len(pending) >= batches_in_flight:
                        future, indices, lengths = pending.popleft()
                        sample_s, sampled = future.result()
                        sampling_call_seconds.append(float(sample_s))
                        write_seconds += _write_sampled_batch(
                            sampled=sampled,
                            target_dir=target_dir,
                            batch_indices=indices,
                            partition_lengths=lengths,
                        )
                        del sampled

                while pending:
                    future, indices, lengths = pending.popleft()
                    sample_s, sampled = future.result()
                    sampling_call_seconds.append(float(sample_s))
                    write_seconds += _write_sampled_batch(
                        sampled=sampled,
                        target_dir=target_dir,
                        batch_indices=indices,
                        partition_lengths=lengths,
                    )
                    del sampled
                sampling_window_stopped = perf_counter()
            finally:
                for future, _, _ in pending:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)

        if rows != expected_rows:
            raise RuntimeError(
                f"Prepared workload contains {rows:,} rows; expected {expected_rows:,}."
            )

    sampling_seconds = float(sum(sampling_call_seconds))
    sampling_window_seconds = (
        None
        if sampling_window_started is None or sampling_window_stopped is None
        else float(sampling_window_stopped - sampling_window_started)
    )
    sampling_overlap_factor = (
        None
        if sampling_window_seconds is None or sampling_window_seconds <= 0
        else sampling_seconds / sampling_window_seconds
    )

    return {
        "rows": rows,
        "partitions": partitions,
        "batches": batches,
        "partitions_per_batch": partitions_per_batch,
        "batches_in_flight": int(batches_in_flight),
        "read_seconds": read_seconds,
        "assembly_seconds": assembly_seconds,
        "geometry_seconds": geometry_seconds,
        "sampling_seconds": sampling_seconds,
        "sampling_call_seconds_max": (
            max(sampling_call_seconds) if sampling_call_seconds else None
        ),
        "sampling_overlap_window_seconds": sampling_window_seconds,
        "sampling_overlap_factor": sampling_overlap_factor,
        "write_seconds": write_seconds,
        "pipeline_seconds": timer["wall_seconds"],
        "operation_memory": timer,
        "write_directory": None if target_dir is None else str(target_dir),
    }


def _base_metadata(
    *,
    args,
    entry,
    raster_workload,
    resource_kind,
    resource_count,
    physical_cores,
    geometry,
    backend,
    memory_gib,
    point_manifest,
):
    point_count = int(entry["target_points"])
    return {
        "campaign": f"point-{args.mode}-scaling-v2",
        "scaling_mode": args.mode,
        "backend": backend,
        "resource_kind": resource_kind,
        "resource_count": resource_count,
        "physical_cores_used": physical_cores,
        "base_geometry": args.geometry,
        "executed_geometry": geometry,
        "point_count": point_count,
        "points_per_physical_core": point_count / physical_cores,
        "point_partition_target_rows": int(point_manifest["partition_size"]),
        "point_partitions": int(entry["partitions"]),
        "point_tile_rows": int(entry["tile_rows"]),
        "point_tile_cols": int(entry["tile_cols"]),
        "point_representation": point_manifest["representation"],
        "point_batches_in_flight": int(args.batches_in_flight),
        "target_raster_gib": raster_workload.target_gib,
        "actual_raster_gib": raster_workload.logical_gib,
        "raster_logical_bytes": raster_workload.logical_bytes,
        "raster_cells": raster_workload.cells,
        "logical_gib_per_physical_core": raster_workload.logical_gib / physical_cores,
        "memory_gib_available_to_campaign": memory_gib,
        "write_sampled_partitions": bool(args.write_sampled),
    }


def _campaign_points(args, *, physical_local, slurm_cores_per_node):
    base_workers, base_threads = parse_geometry(args.geometry)

    if args.backend == "local":
        if physical_local is None:
            raise RuntimeError("Local physical-core count unavailable.")
        if base_workers * base_threads > physical_local:
            raise ValueError(
                f"Geometry {args.geometry} needs {base_workers * base_threads} cores; "
                f"machine has {physical_local}."
            )
        if args.mode == "capacity":
            if not args.point_counts or args.raster_gib is None:
                raise ValueError("capacity requires --point-counts and --raster-gib")
            cores = base_workers * base_threads
            return [
                dict(
                    point_count=count,
                    raster_gib=args.raster_gib,
                    workers=base_workers,
                    threads=base_threads,
                    resource_kind="physical_cores",
                    resource_count=cores,
                    physical_cores=cores,
                )
                for count in args.point_counts
            ]

        if not args.core_counts:
            raise ValueError("local strong/weak scaling requires --core-counts")
        out = []
        for cores in args.core_counts:
            workers, threads = geometry_for_core_budget(args.geometry, cores)
            if args.mode == "strong":
                if args.fixed_points is None or args.raster_gib is None:
                    raise ValueError("strong requires --fixed-points and --raster-gib")
                count = args.fixed_points
                raster_gib = args.raster_gib
            else:
                count = cores * args.points_per_core
                raster_gib = cores * args.gib_per_core
            out.append(
                dict(
                    point_count=count,
                    raster_gib=raster_gib,
                    workers=workers,
                    threads=threads,
                    resource_kind="physical_cores",
                    resource_count=cores,
                    physical_cores=cores,
                )
            )
        return out

    if slurm_cores_per_node is None:
        raise RuntimeError("Slurm physical cores/node unavailable.")
    node_counts = [args.nodes] if args.mode == "capacity" else args.node_counts
    if not node_counts:
        raise ValueError("Slurm strong/weak scaling requires --node-counts")

    out = []
    for nodes in node_counts:
        physical = nodes * slurm_cores_per_node
        if args.mode == "capacity":
            if not args.point_counts or args.raster_gib is None:
                raise ValueError("capacity requires --point-counts and --raster-gib")
            for count in args.point_counts:
                out.append(
                    dict(
                        point_count=count,
                        raster_gib=args.raster_gib,
                        nodes=nodes,
                        resource_kind="nodes",
                        resource_count=nodes,
                        physical_cores=physical,
                    )
                )
            break
        if args.mode == "strong":
            if args.fixed_points is None or args.raster_gib is None:
                raise ValueError("strong requires --fixed-points and --raster-gib")
            count = args.fixed_points
            raster_gib = args.raster_gib
        else:
            count = physical * args.points_per_core
            raster_gib = physical * args.gib_per_core
        out.append(
            dict(
                point_count=count,
                raster_gib=raster_gib,
                nodes=nodes,
                resource_kind="nodes",
                resource_count=nodes,
                physical_cores=physical,
            )
        )
    return out


def _run_repeats(
    *,
    args,
    entry,
    env,
    client,
    output,
    metadata,
    workers,
    threads,
    chunks,
):
    if args.warmup_repeats:
        for warmup in range(1, args.warmup_repeats + 1):
            rows = _warmup_first_tile(
                entry=entry,
                env=env,
                client=client,
                chunks=chunks,
                max_points=args.warmup_points,
            )
            print(
                f"warm-up {warmup}/{args.warmup_repeats}: "
                f"first spatial tile, {rows:,} points"
            )

    target = int(entry["target_points"])
    for repeat in range(1, args.repeats + 1):
        run_key = (
            f"{args.mode}-{metadata['executed_geometry']}-"
            f"{target}points-r{repeat:03d}"
        )
        worker_before = distributed_worker_runtime_snapshot(client)
        node_before = distributed_node_runtime_snapshot(client)
        result = _process_point_workload(
            entry=entry,
            env=env,
            client=client,
            chunks=chunks,
            write_root=args.sampled_output if args.write_sampled else None,
            run_key=run_key,
            partitions_per_batch=args.partitions_per_batch,
            batches_in_flight=args.batches_in_flight,
        )
        node_after = distributed_node_runtime_snapshot(client)
        worker_after = distributed_worker_runtime_snapshot(client)

        telemetry = _telemetry_summary(
            worker_before=worker_before,
            worker_after=worker_after,
            node_before=node_before,
            node_after=node_after,
            wall_seconds=float(result["pipeline_seconds"]),
            workers=workers,
            threads_per_worker=threads,
        )
        shared = {
            **metadata,
            **telemetry,
            "repeat": repeat,
            "warmup_repeats": args.warmup_repeats,
            "warmup_policy": "first_spatial_tile",
            "point_partitions_processed": result["partitions"],
            "point_batches_processed": result["batches"],
            "point_partitions_per_batch": result["partitions_per_batch"],
            "point_batches_in_flight": result["batches_in_flight"],
            "point_read_seconds": result["read_seconds"],
            "point_batch_assembly_seconds": result["assembly_seconds"],
            "point_geometry_seconds": result["geometry_seconds"],
            "point_sampling_seconds": result["sampling_seconds"],
            "point_sampling_call_seconds_max": result["sampling_call_seconds_max"],
            "point_sampling_overlap_window_seconds": result[
                "sampling_overlap_window_seconds"
            ],
            "point_sampling_overlap_factor": result["sampling_overlap_factor"],
            "point_write_seconds": result["write_seconds"],
            "sampled_output_directory": result["write_directory"],
            "computational_chunks": dict(chunks),
        }
        kernel = make_benchmark_record(
            f"point_{args.mode}_sampling_kernel",
            float(result["sampling_seconds"]),
            rows=target,
            workers=workers,
            threads_per_worker=threads,
            chunk_mb=args.chunk_mb,
            metadata={**shared, "stage": "sampling_kernel_sum"},
        )
        pipeline = make_benchmark_record(
            f"point_{args.mode}_pipeline",
            float(result["pipeline_seconds"]),
            rows=target,
            workers=workers,
            threads_per_worker=threads,
            chunk_mb=args.chunk_mb,
            metadata={**shared, "stage": "spatial_parquet_pipeline"},
            operation_memory=result["operation_memory"],
            client=client,
        )
        append_benchmark_record(kernel, output)
        append_benchmark_record(pipeline, output)
        print(
            f"repeat {repeat}: pipeline={pipeline.wall_seconds:.3f}s "
            f"sample_sum={kernel.wall_seconds:.3f}s "
            f"overlap={result['sampling_overlap_factor']:.2f}x "
            f"points/s={pipeline.throughput_rows_s:,.0f}"
        )
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("capacity", "strong", "weak"), required=True)
    parser.add_argument("--backend", choices=("local", "slurm"), required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--point-counts", type=parse_positive_ints, default=None)
    parser.add_argument("--fixed-points", type=int, default=None)
    parser.add_argument("--points-per-core", type=int, default=2_500_000)
    parser.add_argument("--raster-gib", type=float, default=None)
    parser.add_argument("--gib-per-core", type=float, default=2.0)
    parser.add_argument("--core-counts", type=parse_positive_ints, default=None)
    parser.add_argument("--node-counts", type=parse_positive_ints, default=None)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--warmup-points", type=int, default=1_000_000)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--write-sampled", action="store_true")
    parser.add_argument("--sampled-output", type=Path, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--partitions-per-batch",
        type=int,
        default=1,
        help=(
            "Number of spatial Parquet partitions combined into one bounded "
            "sampling call."
        ),
    )
    parser.add_argument(
        "--batches-in-flight",
        type=int,
        default=1,
        help=(
            "Maximum bounded spatial sampling calls allowed to overlap. Default "
            "1 preserves synchronous execution; use a calibrated value for final "
            "campaigns."
        ),
    )
    args = parser.parse_args()

    if args.repeats <= 0 or args.warmup_repeats < 0:
        parser.error("repeats must be positive and warmup-repeats non-negative")
    if min(
        args.points_per_core,
        args.warmup_points,
        args.spatial_chunk,
        args.chunk_mb,
        args.partitions_per_batch,
        args.batches_in_flight,
    ) <= 0:
        parser.error("point/chunk/batch/concurrency sizes must be positive")
    if args.gib_per_core <= 0:
        parser.error("gib-per-core must be positive")
    if args.write_sampled and args.sampled_output is None:
        parser.error("--write-sampled requires --sampled-output")

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.append:
        parser.error(f"{output} already exists; use --append only intentionally")
    if args.sampled_output is not None:
        args.sampled_output = args.sampled_output.expanduser().resolve()
        args.sampled_output.mkdir(parents=True, exist_ok=True)

    full_env, surface_manifest, point_manifest = _load_inputs(root)
    if point_manifest.get("representation") != "spatial-tiled-parquet-v2":
        raise RuntimeError(
            "Point family uses an obsolete representation. Rebuild with "
            "prepare_point_family.py --force."
        )
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", args.spatial_chunk
        )
    )

    physical_local = None
    local_memory_gib = None
    allocation = None
    slurm_cores_per_node = None
    if args.backend == "local":
        physical_local = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        local_memory_gib = psutil.virtual_memory().total / 1024**3
    else:
        allocation = current_slurm_allocation()
        if allocation is None:
            raise RuntimeError("--backend slurm must run inside a Slurm allocation")
        slurm_cores_per_node = _allocation_physical_cores_per_node(allocation)

    campaign_points = _campaign_points(
        args,
        physical_local=physical_local,
        slurm_cores_per_node=slurm_cores_per_node,
    )
    entries = {
        int(point["point_count"]): _point_workload_entry(
            point_manifest, int(point["point_count"])
        )
        for point in campaign_points
    }

    campaign_path = output.with_suffix(output.suffix + ".campaign.json")
    campaign = {
        "campaign": f"point-{args.mode}-scaling-v2",
        "mode": args.mode,
        "backend": args.backend,
        "root": str(root),
        "output": str(output),
        "base_geometry": args.geometry,
        "partitions_per_batch": args.partitions_per_batch,
        "batches_in_flight": args.batches_in_flight,
        "surface_family_manifest": surface_manifest,
        "point_family_manifest": point_manifest,
        "requested_points": campaign_points,
        "host": platform.node(),
        "driver_topology": discover_runtime_topology().as_dict(),
        "status": "running",
    }
    campaign_path.write_text(
        json.dumps(campaign, indent=2, default=str) + "\n", encoding="utf-8"
    )

    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}
    local_tmp = root / "dask-tmp-points"
    local_tmp.mkdir(parents=True, exist_ok=True)

    try:
        for index, point in enumerate(campaign_points, start=1):
            entry = entries[int(point["point_count"])]
            env, raster_workload = _window(
                full_env, float(point["raster_gib"]), storage_chunk
            )
            print(
                f"\n=== point {args.mode} {index}/{len(campaign_points)} ===\n"
                f"points={int(point['point_count']):,} "
                f"tiles={int(entry['partitions'])} "
                f"raster={raster_workload.logical_gib:.3f} GiB "
                f"in_flight={args.batches_in_flight}"
            )

            if args.backend == "local":
                workers = int(point["workers"])
                threads = int(point["threads"])
                geometry = f"{workers}x{threads}"
                execution = _local_configuration(
                    workers=workers,
                    threads=threads,
                    total_memory_gib=float(local_memory_gib),
                    managed_memory_fraction=args.managed_memory_fraction,
                    local_directory=local_tmp,
                    chunk_mb=args.chunk_mb,
                    worker_startup_timeout=args.worker_startup_timeout,
                )
                client, cluster = execution.create_client()
                if client is None:
                    raise RuntimeError("Local point benchmark requires a Dask client.")
                try:
                    client.wait_for_workers(workers, timeout=args.worker_startup_timeout)
                    metadata = _base_metadata(
                        args=args,
                        entry=entry,
                        raster_workload=raster_workload,
                        resource_kind=point["resource_kind"],
                        resource_count=point["resource_count"],
                        physical_cores=point["physical_cores"],
                        geometry=geometry,
                        backend="local",
                        memory_gib=local_memory_gib,
                        point_manifest=point_manifest,
                    )
                    _run_repeats(
                        args=args,
                        entry=entry,
                        env=env,
                        client=client,
                        output=output,
                        metadata=metadata,
                        workers=workers,
                        threads=threads,
                        chunks=chunks,
                    )
                finally:
                    client.close()
                    if cluster is not None:
                        cluster.close()
            else:
                nodes = int(point["nodes"])
                task_count = spatial_task_count(env, chunks)
                plan = coolmuc4_plan(
                    workload="point_sampling",
                    nodes=nodes,
                    cores_per_node=int(slurm_cores_per_node),
                    geometry=args.geometry,
                    task_count=task_count,
                    chunk_mb=args.chunk_mb,
                )
                if allocation.partition == COOLMUC4_POLICY.interactive_partition:
                    plan = replace(plan, partition=allocation.partition)
                elif allocation.partition and allocation.partition != plan.partition:
                    raise RuntimeError(
                        "Current Slurm partition does not match execution plan: "
                        f"{allocation.partition!r} vs {plan.partition!r}"
                    )

                scheduler_file = (
                    output.parent
                    / f"point-scaling-{allocation.job_id}-n{nodes}-{index}.json"
                )
                with slurm_allocation_client(
                    plan,
                    scheduler_file=scheduler_file,
                    interface=args.interface,
                    worker_startup_timeout=args.worker_startup_timeout,
                    validate_topology=False,
                ) as client:
                    validation = validate_worker_topology(
                        client, plan, strict_affinity=True
                    )
                    print(validation.explain())
                    if not validation.ok:
                        raise RuntimeError(
                            "Strict physical-core topology validation failed."
                        )
                    memory_gib = (
                        None
                        if allocation.memory_per_node_gib is None
                        else allocation.memory_per_node_gib * nodes
                    )
                    metadata = {
                        **_base_metadata(
                            args=args,
                            entry=entry,
                            raster_workload=raster_workload,
                            resource_kind=point["resource_kind"],
                            resource_count=point["resource_count"],
                            physical_cores=point["physical_cores"],
                            geometry=plan.geometry.label,
                            backend="slurm",
                            memory_gib=memory_gib,
                            point_manifest=point_manifest,
                        ),
                        "execution_plan": plan.as_dict(),
                        "topology_validation": validation.as_dict(),
                        "slurm_job_id": allocation.job_id,
                        "slurm_partition": allocation.partition,
                        "slurm_node_list": allocation.node_list,
                        "allocation_nodes": allocation.nodes,
                        "node_subset": nodes,
                    }
                    _run_repeats(
                        args=args,
                        entry=entry,
                        env=env,
                        client=client,
                        output=output,
                        metadata=metadata,
                        workers=plan.total_workers,
                        threads=plan.geometry.threads_per_worker,
                        chunks=chunks,
                    )

        campaign["status"] = "completed"
    except BaseException as exc:
        campaign["status"] = "failed"
        campaign["error"] = repr(exc)
        raise
    finally:
        campaign_path.write_text(
            json.dumps(campaign, indent=2, default=str) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()