"""Benchmark block-local extraction for the Dask-native point graph.

The first Dask-native prototype moved Parquet reads and point routing onto Dask
workers and removed the client-side producer bottleneck. Its remaining dominant
telemetry was worker-to-worker transfer into one partition-level extraction task:
that task depended on every candidate raster block touched by the point tile.

This diagnostic keeps worker-side reads/routing but changes the join point:

    parquet read -> route tile -> compact block payload ----\
                                                       block -> extract block
                                                                  |
                                                                  v
                                                        assemble partition

Only compact point positions/flat indices travel toward a raster block. The
large raster block can therefore remain on the worker that materialized it, and
only sampled-value fragments travel onward to the partition assembly task.
Positions and flat raster indices are packed to uint32 whenever possible.

No package behavior is changed.
"""

from __future__ import annotations

import argparse
import gc
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import psutil

from hsa.compute import benchmark_timer, sample_raster_stack_chunked
from hsa.compute.raster import (
    _chunk_number,
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
from profile_point_dask_native_graph import (
    RouteMeta,
    _append_jsonl,
    _candidate_chunk_coords,
    _route_meta,
    _task_events,
    _task_family_summary,
)
from run_point_scaling import (
    _load_inputs,
    _point_frame_to_gdf,
    _point_workload_entry,
)
from run_surface_scaling import _local_configuration, _window


BlockPayload = tuple[np.ndarray, np.ndarray] | None
PartitionMetadata = tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


@dataclass
class BlockLocalRoute:
    """Worker-local routing result aligned to statically known candidate blocks."""

    part_index: int
    rows: int
    point_id: np.ndarray
    used: np.ndarray
    point_x: np.ndarray
    point_y: np.ndarray
    payloads: tuple[BlockPayload, ...]


def _read_partition(path: str) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=["point_id", "u", "v", "used"],
    )


def _route_frame_block_local(
    frame: pd.DataFrame,
    *,
    part_index: int,
    meta: RouteMeta,
    block_coords: tuple[tuple[int, int], ...],
) -> BlockLocalRoute:
    """Route one point tile and align compact payloads to candidate raster blocks."""

    u = frame["u"].to_numpy(dtype=np.float64)
    v = frame["v"].to_numpy(dtype=np.float64)
    point_x = meta.xmin + u * (meta.xmax - meta.xmin)
    point_y = meta.ymin + v * (meta.ymax - meta.ymin)

    row_index = _nearest_indices(meta.y_coordinates, point_y)
    col_index = _nearest_indices(meta.x_coordinates, point_x)
    chunk_y = _chunk_number(row_index, meta.y_lengths)
    chunk_x = _chunk_number(col_index, meta.x_lengths)
    grouped = _group_positions_by_chunk(
        chunk_y,
        chunk_x,
        n_x_chunks=len(meta.x_lengths),
    )

    position_dtype = (
        np.uint32 if len(frame) <= np.iinfo(np.uint32).max else np.uint64
    )
    payload_map: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for cy, cx, positions in grouped:
        y_start = int(meta.y_starts[cy])
        x_start = int(meta.x_starts[cx])
        block_width = int(meta.x_lengths[cx])
        local_rows = row_index[positions] - y_start
        local_cols = col_index[positions] - x_start
        max_flat_index = int(meta.y_lengths[cy]) * block_width - 1
        flat_dtype = (
            np.uint32
            if max_flat_index <= np.iinfo(np.uint32).max
            else np.uint64
        )
        packed_positions = positions.astype(position_dtype, copy=False)
        flat_indices = (
            local_rows * block_width + local_cols
        ).astype(flat_dtype, copy=False)
        payload_map[(int(cy), int(cx))] = (packed_positions, flat_indices)

    unexpected = set(payload_map).difference(block_coords)
    if unexpected:
        raise RuntimeError(
            f"Tile {part_index} routed outside static candidate blocks: "
            f"{sorted(unexpected)!r}."
        )

    return BlockLocalRoute(
        part_index=int(part_index),
        rows=len(frame),
        point_id=frame["point_id"].to_numpy(dtype=np.int64, copy=True),
        used=frame["used"].to_numpy(dtype=bool, copy=True),
        point_x=point_x,
        point_y=point_y,
        payloads=tuple(payload_map.get(coord) for coord in block_coords),
    )


def _select_partition_metadata(routed: BlockLocalRoute) -> PartitionMetadata:
    """Project the routing result down to the fields needed by final assembly."""

    return (
        routed.part_index,
        routed.rows,
        routed.point_id,
        routed.used,
        routed.point_x,
        routed.point_y,
    )


def _select_block_payload(
    routed: BlockLocalRoute,
    payload_index: int,
) -> BlockPayload:
    """Expose only one compact block payload to a downstream extraction task."""

    return routed.payloads[payload_index]


def _extract_block_local(
    payload: BlockPayload,
    block,
    dtype: str,
):
    """Sample one raster block where it is materialized.

    The dependency on ``block`` is intentionally local to this task. Dask's
    placement heuristic can therefore move the much smaller routing payload to
    the raster block rather than moving every raster block to one partition task.
    """

    if payload is None:
        return None
    positions, flat_indices = payload
    extracted = _extract_numpy_block_flat(
        np.asarray(block),
        flat_indices,
    ).astype(np.dtype(dtype), copy=False)
    return positions, extracted


def _assemble_block_local_partition(
    metadata: PartitionMetadata,
    sampled_bands: tuple[str, ...],
    dtype: str,
    *fragments,
) -> pd.DataFrame:
    """Assemble sampled block fragments into production-compatible row order."""

    part_index, rows, point_id, used, point_x, point_y = metadata
    out_dtype = np.dtype(dtype)
    values = np.empty((len(sampled_bands), rows), dtype=out_dtype)
    filled = 0

    for fragment in fragments:
        if fragment is None:
            continue
        positions, extracted = fragment
        positions = np.asarray(positions)
        extracted = np.asarray(extracted, dtype=out_dtype)
        values[:, positions] = extracted
        filled += len(positions)

    if filled != rows:
        raise RuntimeError(
            f"Tile {part_index} assembled {filled:,} sampled positions; "
            f"expected {rows:,}."
        )

    out = pd.DataFrame(values.T, columns=list(sampled_bands))
    out["x"] = point_x
    out["y"] = point_y
    out["used"] = used
    out["point_id"] = point_id
    return out


def _partition_delayed(
    *,
    workload_dir: Path,
    part_index: int,
    tile_rows: int,
    tile_cols: int,
    meta: RouteMeta,
    raster_blocks,
):
    path = workload_dir / f"part-{part_index:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing point tile: {path}")

    coords = _candidate_chunk_coords(
        part_index=part_index,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        meta=meta,
    )
    frame = dask.delayed(_read_partition, pure=False)(str(path))
    routed = dask.delayed(_route_frame_block_local, pure=False)(
        frame,
        part_index=part_index,
        meta=meta,
        block_coords=coords,
    )
    metadata = dask.delayed(_select_partition_metadata, pure=False)(routed)

    fragments = []
    for payload_index, (cy, cx) in enumerate(coords):
        payload = dask.delayed(_select_block_payload, pure=False)(
            routed,
            payload_index,
        )
        block = raster_blocks[0, cy, cx]
        fragment = dask.delayed(_extract_block_local, pure=False)(
            payload,
            block,
            meta.dtype,
        )
        fragments.append(fragment)

    sampled = dask.delayed(_assemble_block_local_partition, pure=False)(
        metadata,
        meta.sampled_bands,
        meta.dtype,
        *fragments,
    )
    return sampled, len(coords)


def _validate_candidate(
    *,
    entry,
    env,
    chunks,
    meta: RouteMeta,
    raster_blocks,
    tile_rows: int,
    tile_cols: int,
    client,
    validation_points: int,
) -> None:
    path = Path(entry["directory"]) / "part-000000.parquet"
    frame = pd.read_parquet(
        path,
        columns=["point_id", "u", "v", "used"],
    )
    if len(frame) > validation_points:
        frame = frame.iloc[:validation_points].copy()

    points = _point_frame_to_gdf(frame, env)
    reference = sample_raster_stack_chunked(
        points,
        env,
        chunks=chunks,
        client=client,
        id_cols="point_id",
        require_inside=True,
    )

    coords = _candidate_chunk_coords(
        part_index=0,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        meta=meta,
    )
    routed = dask.delayed(_route_frame_block_local, pure=False)(
        frame,
        part_index=0,
        meta=meta,
        block_coords=coords,
    )
    metadata = dask.delayed(_select_partition_metadata, pure=False)(routed)
    fragments = []
    for payload_index, (cy, cx) in enumerate(coords):
        payload = dask.delayed(_select_block_payload, pure=False)(
            routed,
            payload_index,
        )
        fragments.append(
            dask.delayed(_extract_block_local, pure=False)(
                payload,
                raster_blocks[0, cy, cx],
                meta.dtype,
            )
        )
    candidate_task = dask.delayed(_assemble_block_local_partition, pure=False)(
        metadata,
        meta.sampled_bands,
        meta.dtype,
        *fragments,
    )
    candidate = client.compute(candidate_task).result()
    candidate.index = reference.index
    pd.testing.assert_frame_equal(reference, candidate, check_exact=True)

    del frame, points, reference, candidate
    gc.collect()


def _run_once(
    *,
    entry,
    meta: RouteMeta,
    raster_blocks,
    client,
    tile_rows: int,
    tile_cols: int,
    submission_group_partitions: int,
    groups_in_flight: int,
    workers: int,
    threads_per_worker: int,
) -> dict[str, Any]:
    workload_dir = Path(entry["directory"])
    expected_partitions = int(entry["partitions"])
    expected_rows = int(entry["target_points"])

    worker_before = distributed_worker_runtime_snapshot(client)
    driver_before = process_runtime_snapshot()
    node_before = distributed_node_runtime_snapshot(client)
    task_start = time.time()

    rows = 0
    candidate_block_dependencies = 0
    peak_pending_groups = 0
    pending: deque[list[Any]] = deque()

    with benchmark_timer(client=client) as timer:
        for group_start in range(
            0,
            expected_partitions,
            submission_group_partitions,
        ):
            group_stop = min(
                group_start + submission_group_partitions,
                expected_partitions,
            )
            tasks = []
            for part_index in range(group_start, group_stop):
                task, block_count = _partition_delayed(
                    workload_dir=workload_dir,
                    part_index=part_index,
                    tile_rows=tile_rows,
                    tile_cols=tile_cols,
                    meta=meta,
                    raster_blocks=raster_blocks,
                )
                tasks.append(task)
                candidate_block_dependencies += block_count

            futures = list(client.compute(tasks))
            pending.append(futures)
            peak_pending_groups = max(peak_pending_groups, len(pending))

            if len(pending) >= groups_in_flight:
                oldest = pending.popleft()
                computed = client.gather(oldest)
                rows += sum(len(frame) for frame in computed)
                client.cancel(oldest)
                del computed, oldest
                gc.collect()

        while pending:
            futures = pending.popleft()
            computed = client.gather(futures)
            rows += sum(len(frame) for frame in computed)
            client.cancel(futures)
            del computed, futures
            gc.collect()

    task_stop = time.time()
    node_after = distributed_node_runtime_snapshot(client)
    driver_after = process_runtime_snapshot()
    worker_after = distributed_worker_runtime_snapshot(client)
    events = _task_events(client, start=task_start, stop=task_stop)

    if rows != expected_rows:
        raise RuntimeError(
            f"Block-local graph returned {rows:,} rows; expected {expected_rows:,}."
        )

    wall = float(timer["wall_seconds"])
    worker = aggregate_worker_runtime_delta(worker_before, worker_after)
    driver = process_runtime_delta(driver_before, driver_after)
    nodes = distributed_node_runtime_delta(node_before, node_after)
    tasks = summarize_task_stream(events)
    families = _task_family_summary(events)
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
        "submission_group_partitions": int(submission_group_partitions),
        "groups_in_flight": int(groups_in_flight),
        "peak_pending_groups": int(peak_pending_groups),
        "candidate_block_dependencies": int(candidate_block_dependencies),
        "candidate_blocks_mean_per_partition": (
            candidate_block_dependencies / expected_partitions
        ),
        "block_local_extraction_tasks": int(candidate_block_dependencies),
        "worker_cpu_seconds": worker_cpu,
        "worker_busy_cores_pipeline": worker_busy,
        "worker_execution_thread_utilization_fraction": (
            None
            if worker_busy is None
            else worker_busy / (workers * threads_per_worker)
        ),
        "driver_cpu_seconds": driver_cpu,
        "driver_busy_cores_pipeline": driver_busy,
        "task_stream_tasks": tasks.get("tasks"),
        "task_compute_seconds": tasks.get("compute_seconds"),
        "task_transfer_seconds": tasks.get("transfer_seconds"),
        "task_deserialize_seconds": tasks.get("deserialize_seconds"),
        "task_compute_parallelism": tasks.get("compute_parallelism"),
        "task_compute_parallelism_fraction_of_execution_threads": (
            None
            if tasks.get("compute_parallelism") is None
            else tasks["compute_parallelism"] / (workers * threads_per_worker)
        ),
        "task_stream_span_seconds": tasks.get("span_seconds"),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark block-local Dask-native point extraction."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--geometry", default="6x2")
    parser.add_argument(
        "--submission-group-partitions",
        default="16,32,64",
        help="Comma-separated point partitions submitted per static Dask graph.",
    )
    parser.add_argument("--groups-in-flight", type=int, default=2)
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
        args.groups_in_flight,
        args.spatial_chunk,
        args.chunk_mb,
        args.validation_points,
        args.repeats,
    ) <= 0:
        parser.error("all point/group/chunk/repeat values must be positive")

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
    tile_rows = int(entry["tile_rows"])
    tile_cols = int(entry["tile_cols"])
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk",
            args.spatial_chunk,
        )
    )
    env, raster_workload = _window(full_env, args.raster_gib, storage_chunk)
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}
    meta, raster = _route_meta(env, chunks)
    raster_blocks = np.asarray(raster.data.to_delayed(), dtype=object)
    if raster_blocks.ndim != 3 or raster_blocks.shape[0] != 1:
        raise RuntimeError(
            "Expected one band chunk and a 2-D spatial raster block grid; "
            f"observed delayed block shape {raster_blocks.shape}."
        )

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
        raise RuntimeError("Block-local point profile requires a Dask client.")

    try:
        client.wait_for_workers(workers, timeout=args.worker_startup_timeout)
        try:
            client.get_task_stream()
        except Exception:
            pass

        _validate_candidate(
            entry=entry,
            env=env,
            chunks=chunks,
            meta=meta,
            raster_blocks=raster_blocks,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            client=client,
            validation_points=args.validation_points,
        )
        print(f"validation passed for {args.validation_points:,} points")

        for repeat in range(1, args.repeats + 1):
            ordered = group_sizes if repeat % 2 else list(reversed(group_sizes))
            for group_size in ordered:
                record = _run_once(
                    entry=entry,
                    meta=meta,
                    raster_blocks=raster_blocks,
                    client=client,
                    tile_rows=tile_rows,
                    tile_cols=tile_cols,
                    submission_group_partitions=group_size,
                    groups_in_flight=args.groups_in_flight,
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
                        "architecture": "dask_native_block_local",
                    }
                )
                _append_jsonl(output, record)
                print(
                    f"repeat {repeat} group={group_size}: "
                    f"pipeline={record['pipeline_seconds']:.3f}s "
                    f"points/s={record['throughput_points_s']:,.0f} "
                    f"worker_cores={record['worker_busy_cores_pipeline']} "
                    f"task_parallelism={record['task_compute_parallelism']} "
                    f"transfer_s={record['task_transfer_seconds']} "
                    f"driver_cores={record['driver_busy_cores_pipeline']}"
                )
                gc.collect()
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
