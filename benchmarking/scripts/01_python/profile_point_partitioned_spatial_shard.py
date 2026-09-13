"""Benchmark one spatial shard of the partition-native point sampler.

This runner is intended to be launched inside a socket-sized CPU affinity mask.
The complete prepared x/y point cache is reused, point tiles are split into
deterministic vertical bands from the prepared tile grid, and the raster is
cropped to a chunk-aligned window covering only the selected tiles.

Each shard creates a local Dask cluster inside its inherited affinity mask and
pins workers to non-overlapping physical-core subsets.  An optional filesystem
barrier synchronizes the start of timed sampling across concurrent shards.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from hsa.compute import PointPartition
from profile_point_partitioned_production import (
    _append_jsonl,
    _materialize_xy_cache,
    _run_once,
    _validate_production_path,
)
from run_point_scaling import _load_inputs, _point_workload_entry
from run_surface_scaling import _window


def _task_rank(explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    for name in ("SLURM_PROCID", "SLURM_LOCALID"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def _column_shard_indices(
    *,
    entry: dict[str, Any],
    partition_count: int,
    shard_index: int,
    shard_count: int,
) -> tuple[list[int], tuple[int, int]]:
    """Return prepared tile indices belonging to one vertical spatial band."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")

    tile_rows = int(entry.get("tile_rows", 0))
    tile_cols = int(entry.get("tile_cols", 0))
    if tile_rows <= 0 or tile_cols <= 0:
        raise RuntimeError("Point workload manifest has no valid tile_rows/tile_cols.")
    if tile_rows * tile_cols != int(partition_count):
        raise RuntimeError(
            "Prepared tile grid does not match partition count: "
            f"{tile_rows}x{tile_cols} != {partition_count}."
        )

    col_start = (shard_index * tile_cols) // shard_count
    col_stop = ((shard_index + 1) * tile_cols) // shard_count
    indices = [
        index
        for index in range(partition_count)
        if col_start <= (index % tile_cols) < col_stop
    ]
    if not indices:
        raise RuntimeError(
            f"Spatial shard {shard_index}/{shard_count} contains no point tiles."
        )
    return indices, (col_start, col_stop)


def _covering_index_bounds(
    coordinates: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[int, int]:
    """Return index bounds whose coordinate centres bracket ``[lower, upper]``.

    Raster sampling with ``require_inside=True`` treats the minimum and maximum
    coordinate *centres* as the valid extent.  Choosing merely the nearest raster
    index to a point bound can therefore crop half a pixel too tightly: points on
    the outer half of that nearest cell then lie beyond the cropped coordinate
    extent even though they are valid in the full raster.  Bracketing with a
    centre on each side preserves the full-raster inside test after cropping.

    Both ascending and descending regular coordinate vectors are supported.
    """
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("coordinates must be a non-empty one-dimensional vector")

    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    ascending = bool(values[0] <= values[-1])
    ordered = values if ascending else values[::-1]

    if lo < float(ordered[0]) or hi > float(ordered[-1]):
        raise ValueError(
            "point bounds fall outside the full environmental raster coordinate extent"
        )

    lower_ordered = max(0, int(np.searchsorted(ordered, lo, side="right")) - 1)
    upper_ordered = min(
        int(ordered.size) - 1,
        int(np.searchsorted(ordered, hi, side="left")),
    )

    if ascending:
        first, last = lower_ordered, upper_ordered
    else:
        first = int(values.size) - 1 - upper_ordered
        last = int(values.size) - 1 - lower_ordered
    return min(first, last), max(first, last)


def _aligned_raster_window(
    env,
    partitions: list[PointPartition],
    *,
    spatial_chunk: int,
):
    """Crop ``env`` to chunk-aligned cells covering the selected point bounds."""
    if spatial_chunk <= 0:
        raise ValueError("spatial_chunk must be positive")
    if not partitions:
        raise ValueError("partitions must not be empty")

    xmin = min(float(part.bounds[0]) for part in partitions)
    ymin = min(float(part.bounds[1]) for part in partitions)
    xmax = max(float(part.bounds[2]) for part in partitions)
    ymax = max(float(part.bounds[3]) for part in partitions)

    x = np.asarray(env["x"].values, dtype=np.float64)
    y = np.asarray(env["y"].values, dtype=np.float64)
    x_min, x_max = _covering_index_bounds(x, xmin, xmax)
    y_min, y_max = _covering_index_bounds(y, ymin, ymax)

    x0 = (x_min // spatial_chunk) * spatial_chunk
    y0 = (y_min // spatial_chunk) * spatial_chunk
    x1 = min(
        int(env.sizes["x"]),
        ((x_max + 1 + spatial_chunk - 1) // spatial_chunk) * spatial_chunk,
    )
    y1 = min(
        int(env.sizes["y"]),
        ((y_max + 1 + spatial_chunk - 1) // spatial_chunk) * spatial_chunk,
    )

    cropped = env.isel(x=slice(x0, x1), y=slice(y0, y1))
    return cropped, {
        "x_start": x0,
        "x_stop": x1,
        "y_start": y0,
        "y_stop": y1,
        "x_cells": x1 - x0,
        "y_cells": y1 - y0,
        "point_bounds": [xmin, ymin, xmax, ymax],
    }


def _cpu_topology(cpu_id: int) -> tuple[int, int]:
    base = Path(f"/sys/devices/system/cpu/cpu{int(cpu_id)}/topology")
    try:
        package = int((base / "physical_package_id").read_text().strip())
        core = int((base / "core_id").read_text().strip())
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot resolve physical topology for CPU {cpu_id}.") from exc
    return package, core


def _physical_core_groups(cpu_ids: list[int]) -> list[list[int]]:
    """Group logical CPU ids by physical package/core within the task cpuset."""
    grouped: dict[tuple[int, int], list[int]] = {}
    for cpu_id in cpu_ids:
        grouped.setdefault(_cpu_topology(cpu_id), []).append(int(cpu_id))
    return [sorted(grouped[key]) for key in sorted(grouped)]


def _set_process_affinity(cpu_ids: list[int]) -> list[int]:
    os.sched_setaffinity(0, set(int(cpu) for cpu in cpu_ids))
    return sorted(int(cpu) for cpu in os.sched_getaffinity(0))


def _pin_workers(
    client,
    *,
    task_core_groups: list[list[int]],
    workers: int,
    threads: int,
):
    needed = workers * threads
    if len(task_core_groups) < needed:
        raise RuntimeError(
            f"Shard process exposes {len(task_core_groups)} physical cores but geometry needs {needed}."
        )
    addresses = sorted(client.scheduler_info().get("workers", {}))
    if len(addresses) != workers:
        raise RuntimeError(f"Expected {workers} Dask workers, observed {len(addresses)}.")

    assignments: dict[str, list[int]] = {}
    for index, address in enumerate(addresses):
        physical_groups = task_core_groups[index * threads : (index + 1) * threads]
        cpus = sorted(cpu for group in physical_groups for cpu in group)
        observed = client.run(_set_process_affinity, cpus, workers=[address])
        actual = sorted(list(observed.values())[0])
        if actual != cpus:
            raise RuntimeError(
                f"Worker {address} affinity mismatch: requested {cpus}, observed {actual}."
            )
        assignments[address] = cpus
    return assignments


def _wait_for_barrier(
    barrier_dir: Path,
    *,
    shard_index: int,
    shard_count: int,
    timeout: float,
) -> None:
    """Synchronize timed sampling starts across independently launched shards."""
    barrier_dir.mkdir(parents=True, exist_ok=True)
    ready = barrier_dir / f"ready-{shard_index:02d}"
    ready.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + float(timeout)
    while True:
        ready_count = len(list(barrier_dir.glob("ready-*")))
        if ready_count >= shard_count:
            print(
                f"shard {shard_index}: timing barrier released ({ready_count}/{shard_count})",
                flush=True,
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Shard {shard_index} timed out waiting for {shard_count} ready shards."
            )
        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one socket-local spatial point-sampling shard."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=1_000_000_000)
    parser.add_argument("--raster-gib", type=float, default=192.0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=14)
    parser.add_argument("--graph-partitions", type=int, default=128)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--managed-memory-gib", type=float, default=96.0)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--barrier-dir", type=Path, default=None)
    parser.add_argument("--barrier-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if min(
        args.point_count,
        args.shard_count,
        args.workers,
        args.threads_per_worker,
        args.graph_partitions,
        args.spatial_chunk,
        args.validation_points,
        args.repeats,
    ) <= 0:
        parser.error("point/shard/worker/thread/graph/chunk/validation/repeat values must be positive")
    if args.raster_gib <= 0 or args.managed_memory_gib <= 0 or args.barrier_timeout <= 0:
        parser.error("raster-gib, managed-memory-gib and barrier-timeout must be positive")

    shard_index = _task_rank(args.shard_index)
    if not 0 <= shard_index < args.shard_count:
        parser.error(
            f"Resolved shard index {shard_index} outside [0, {args.shard_count})."
        )

    task_cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    task_core_groups = _physical_core_groups(task_cpus)
    required_physical_cores = args.workers * args.threads_per_worker
    if len(task_core_groups) != required_physical_cores:
        raise RuntimeError(
            "Socket-shard benchmark expects the process to expose exactly "
            f"{required_physical_cores} physical cores; observed "
            f"{len(task_core_groups)} physical cores across {len(task_cpus)} logical CPUs."
        )

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"shard-{shard_index:02d}.jsonl"
    output.unlink(missing_ok=True)

    full_env, surface_manifest, point_manifest = _load_inputs(root)
    entry = _point_workload_entry(point_manifest, args.point_count)
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", args.spatial_chunk
        )
    )
    env, raster_workload = _window(full_env, args.raster_gib, storage_chunk)

    # Reuse the *full-window* x/y cache. Mapping u/v directly into a shard crop
    # would change the benchmark point coordinates.
    all_partitions = _materialize_xy_cache(root, entry, env)
    selected_indices, tile_columns = _column_shard_indices(
        entry=entry,
        partition_count=len(all_partitions),
        shard_index=shard_index,
        shard_count=args.shard_count,
    )
    partitions = [all_partitions[index] for index in selected_indices]
    shard_env, raster_window = _aligned_raster_window(
        env,
        partitions,
        spatial_chunk=args.spatial_chunk,
    )
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}

    from distributed import Client, LocalCluster

    local_root = Path(os.environ.get("TMPDIR", output_dir)) / (
        f"hrhsa-point-shard-{os.environ.get('SLURM_JOB_ID', 'local')}-{shard_index}"
    )
    local_root.mkdir(parents=True, exist_ok=True)
    memory_limit_bytes = int(args.managed_memory_gib * 1024**3 / args.workers)
    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
        memory_limit=memory_limit_bytes,
        local_directory=str(local_root),
        dashboard_address=None,
    )
    client = Client(cluster)

    try:
        client.wait_for_workers(args.workers, timeout=args.worker_startup_timeout)
        worker_affinity = _pin_workers(
            client,
            task_core_groups=task_core_groups,
            workers=args.workers,
            threads=args.threads_per_worker,
        )
        print(
            f"shard {shard_index}/{args.shard_count}: "
            f"tiles={len(partitions)} rows={sum(p.rows or 0 for p in partitions):,} "
            f"tile_cols={tile_columns[0]}:{tile_columns[1]} "
            f"raster={int(shard_env.sizes['y'])}x{int(shard_env.sizes['x'])} "
            f"physical_cores={len(task_core_groups)} logical_cpus={len(task_cpus)}",
            flush=True,
        )
        print("task logical CPU affinity:", task_cpus, flush=True)
        print("worker affinity:", json.dumps(worker_affinity, sort_keys=True), flush=True)

        try:
            client.get_task_stream()
        except Exception:
            pass

        _validate_production_path(
            partitions=partitions,
            env=shard_env,
            client=client,
            chunks=chunks,
            validation_points=min(args.validation_points, int(partitions[0].rows or 0)),
        )

        if args.barrier_dir is not None:
            _wait_for_barrier(
                args.barrier_dir.expanduser().resolve(),
                shard_index=shard_index,
                shard_count=args.shard_count,
                timeout=args.barrier_timeout,
            )

        for repeat in range(1, args.repeats + 1):
            record = _run_once(
                partitions=partitions,
                env=shard_env,
                client=client,
                chunks=chunks,
                graph_partitions=min(args.graph_partitions, len(partitions)),
                workers=args.workers,
                threads_per_worker=args.threads_per_worker,
            )
            record.update(
                {
                    "repeat": repeat,
                    "backend": "slurm-socket-localcluster",
                    "architecture": "production_partitioned_block_local_spatial_shard",
                    "shard_index": shard_index,
                    "shard_count": args.shard_count,
                    "tile_column_start": tile_columns[0],
                    "tile_column_stop": tile_columns[1],
                    "partition_index_min": min(selected_indices),
                    "partition_index_max": max(selected_indices),
                    "workers": args.workers,
                    "threads_per_worker": args.threads_per_worker,
                    "execution_threads": required_physical_cores,
                    "task_logical_cpu_ids": task_cpus,
                    "task_physical_core_count": len(task_core_groups),
                    "worker_affinity": worker_affinity,
                    "point_count_full": int(args.point_count),
                    "target_raster_gib_full": float(args.raster_gib),
                    "raster_gib_full": float(raster_workload.logical_gib),
                    "raster_gib_shard": float(shard_env.nbytes) / 1024**3,
                    "raster_window": raster_window,
                    "spatial_chunk": args.spatial_chunk,
                    "managed_memory_gib": args.managed_memory_gib,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
                }
            )
            _append_jsonl(output, record)
            print(
                f"shard={shard_index} repeat={repeat} rows={record['rows']:,} "
                f"pipeline={record['pipeline_seconds']:.3f}s "
                f"points/s={record['throughput_points_s']:,.0f} "
                f"worker_cores={record['worker_busy_cores_pipeline']} "
                f"task_parallelism={record['task_compute_parallelism']} "
                f"rss_mb={record['operation_peak_process_tree_rss_mb']}",
                flush=True,
            )
            gc.collect()
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
