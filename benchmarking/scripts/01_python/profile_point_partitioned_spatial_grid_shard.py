"""Benchmark one balanced 2-D spatial shard of partition-native point sampling.

This is the grid-sharded companion to ``profile_point_partitioned_spatial_shard``.
Instead of splitting only along point-tile columns, it divides the prepared
row/column tile grid into rectangular shard cells.  That keeps spatial locality
while allowing finer shard counts such as 16 to remain well balanced.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

from distributed import Client, LocalCluster

from profile_point_partitioned_production import (
    _append_jsonl,
    _materialize_xy_cache,
    _run_once,
    _validate_production_path,
)
from profile_point_partitioned_spatial_shard import (
    _aligned_raster_window,
    _physical_core_groups,
    _pin_workers,
    _wait_for_barrier,
)
from run_point_scaling import _load_inputs, _point_workload_entry
from run_surface_scaling import _window


def _grid_shard_indices(
    *,
    entry: dict[str, Any],
    partition_count: int,
    shard_index: int,
    shard_rows: int,
    shard_cols: int,
) -> tuple[list[int], dict[str, int]]:
    tile_rows = int(entry.get("tile_rows", 0))
    tile_cols = int(entry.get("tile_cols", 0))
    if tile_rows <= 0 or tile_cols <= 0:
        raise RuntimeError("Point workload manifest has no valid tile_rows/tile_cols.")
    if tile_rows * tile_cols != int(partition_count):
        raise RuntimeError(
            f"Prepared tile grid {tile_rows}x{tile_cols} does not match "
            f"partition count {partition_count}."
        )
    if shard_rows <= 0 or shard_cols <= 0:
        raise ValueError("shard_rows and shard_cols must be positive")
    shard_count = shard_rows * shard_cols
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must lie in [0, {shard_count})")

    shard_row, shard_col = divmod(shard_index, shard_cols)
    row_start = (shard_row * tile_rows) // shard_rows
    row_stop = ((shard_row + 1) * tile_rows) // shard_rows
    col_start = (shard_col * tile_cols) // shard_cols
    col_stop = ((shard_col + 1) * tile_cols) // shard_cols

    indices = [
        row * tile_cols + col
        for row in range(row_start, row_stop)
        for col in range(col_start, col_stop)
    ]
    if not indices:
        raise RuntimeError(
            f"Grid shard {shard_index}/{shard_count} contains no point tiles."
        )
    return indices, {
        "tile_row_start": row_start,
        "tile_row_stop": row_stop,
        "tile_col_start": col_start,
        "tile_col_stop": col_stop,
        "tile_rows": row_stop - row_start,
        "tile_cols": col_stop - col_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one balanced rectangular spatial point shard."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=1_000_000_000)
    parser.add_argument("--raster-gib", type=float, default=192.0)
    parser.add_argument("--shard-rows", type=int, required=True)
    parser.add_argument("--shard-cols", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, required=True)
    parser.add_argument("--graph-partitions", type=int, required=True)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--managed-memory-gib", type=float, required=True)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--barrier-dir", type=Path, required=True)
    parser.add_argument("--barrier-timeout", type=float, default=600.0)
    args = parser.parse_args()

    shard_count = args.shard_rows * args.shard_cols
    if min(
        args.point_count,
        args.shard_rows,
        args.shard_cols,
        args.workers,
        args.threads_per_worker,
        args.graph_partitions,
        args.spatial_chunk,
        args.validation_points,
        args.repeats,
    ) <= 0:
        parser.error("all count/chunk/thread/graph values must be positive")
    if not 0 <= args.shard_index < shard_count:
        parser.error(f"shard-index must lie in [0, {shard_count})")

    task_cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    task_core_groups = _physical_core_groups(task_cpus)
    required_physical = args.workers * args.threads_per_worker
    if len(task_core_groups) != required_physical:
        raise RuntimeError(
            f"Grid shard expects {required_physical} physical cores; observed "
            f"{len(task_core_groups)} across {len(task_cpus)} logical CPUs."
        )

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"shard-{args.shard_index:02d}.jsonl"
    output.unlink(missing_ok=True)

    full_env, surface_manifest, point_manifest = _load_inputs(root)
    entry = _point_workload_entry(point_manifest, args.point_count)
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", args.spatial_chunk
        )
    )
    env, raster_workload = _window(full_env, args.raster_gib, storage_chunk)
    all_partitions = _materialize_xy_cache(root, entry, env)
    selected_indices, tile_window = _grid_shard_indices(
        entry=entry,
        partition_count=len(all_partitions),
        shard_index=args.shard_index,
        shard_rows=args.shard_rows,
        shard_cols=args.shard_cols,
    )
    partitions = [all_partitions[index] for index in selected_indices]
    shard_env, raster_window = _aligned_raster_window(
        env,
        partitions,
        spatial_chunk=args.spatial_chunk,
    )
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}

    local_root = Path(os.environ.get("TMPDIR", output_dir)) / (
        f"hrhsa-point-grid-{os.environ.get('SLURM_JOB_ID', 'local')}-{args.shard_index}"
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
        rows = sum(int(part.rows or 0) for part in partitions)
        print(
            f"grid shard {args.shard_index}/{shard_count}: tiles={len(partitions)} "
            f"rows={rows:,} tile_window={tile_window} "
            f"raster={int(shard_env.sizes['y'])}x{int(shard_env.sizes['x'])} "
            f"physical_cores={len(task_core_groups)} logical_cpus={len(task_cpus)}",
            flush=True,
        )
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

        _wait_for_barrier(
            args.barrier_dir.expanduser().resolve(),
            shard_index=args.shard_index,
            shard_count=shard_count,
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
                    "backend": "slurm-localcluster-grid-shard",
                    "architecture": "production_partitioned_block_local_spatial_grid_shard",
                    "shard_index": args.shard_index,
                    "shard_count": shard_count,
                    "shard_rows": args.shard_rows,
                    "shard_cols": args.shard_cols,
                    **tile_window,
                    "partition_index_min": min(selected_indices),
                    "partition_index_max": max(selected_indices),
                    "workers": args.workers,
                    "threads_per_worker": args.threads_per_worker,
                    "execution_threads": required_physical,
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
                }
            )
            _append_jsonl(output, record)
            print(
                f"grid_shard={args.shard_index} repeat={repeat} rows={record['rows']:,} "
                f"pipeline={record['pipeline_seconds']:.3f}s "
                f"points/s={record['throughput_points_s']:,.0f} "
                f"worker_cores={record['worker_busy_cores_pipeline']} "
                f"task_parallelism={record['task_compute_parallelism']}",
                flush=True,
            )
            gc.collect()
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
