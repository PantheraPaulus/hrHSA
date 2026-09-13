"""Benchmark one workload-balanced spatial point shard.

Shards are planned from PointPartition row counts and spatial bounds using
``hsa.compute.plan_spatial_point_shards``.  Unlike a regular grid, the planner
can create mildly irregular boundaries so arbitrary shard counts remain tightly
balanced while preserving compact raster windows.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

from distributed import Client, LocalCluster

from hsa.compute import plan_spatial_point_shards
from profile_point_partitioned_production import (
    _append_jsonl,
    _materialize_xy_cache,
    _run_once,
    _validate_production_path,
    _xy_cache_directory,
)
from profile_point_partitioned_spatial_shard import (
    _aligned_raster_window,
    _physical_core_groups,
    _pin_workers,
    _wait_for_barrier,
)
from run_point_scaling import _load_inputs, _point_workload_entry
from run_surface_scaling import _window


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one workload-balanced spatial point shard."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=1_000_000_000)
    parser.add_argument("--raster-gib", type=float, default=192.0)
    parser.add_argument("--shard-count", type=int, required=True)
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
        parser.error("all count/chunk/thread/graph values must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error(f"shard-index must lie in [0, {args.shard_count})")

    process_started = time.perf_counter()
    task_cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    task_core_groups = _physical_core_groups(task_cpus)
    required_physical = args.workers * args.threads_per_worker
    if len(task_core_groups) != required_physical:
        raise RuntimeError(
            f"Balanced shard expects {required_physical} physical cores; observed "
            f"{len(task_core_groups)} across {len(task_cpus)} logical CPUs."
        )

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"shard-{args.shard_index:02d}.jsonl"
    setup_output = output_dir / f"shard-{args.shard_index:02d}.setup.json"
    output.unlink(missing_ok=True)
    setup_output.unlink(missing_ok=True)

    phase = time.perf_counter()
    full_env, surface_manifest, point_manifest = _load_inputs(root)
    entry = _point_workload_entry(point_manifest, args.point_count)
    input_load_seconds = time.perf_counter() - phase

    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", args.spatial_chunk
        )
    )
    phase = time.perf_counter()
    env, raster_workload = _window(full_env, args.raster_gib, storage_chunk)
    raster_window_selection_seconds = time.perf_counter() - phase

    cache_dir = _xy_cache_directory(root, entry, env)
    xy_cache_present_before = (cache_dir / "manifest.json").exists()
    phase = time.perf_counter()
    all_partitions = _materialize_xy_cache(root, entry, env)
    xy_cache_seconds = time.perf_counter() - phase

    phase = time.perf_counter()
    plan = plan_spatial_point_shards(all_partitions, args.shard_count)
    shard_plan_seconds = time.perf_counter() - phase
    shard = plan[args.shard_index]
    selected_indices = list(shard.partition_indices)
    partitions = [all_partitions[index] for index in selected_indices]

    phase = time.perf_counter()
    shard_env, raster_window = _aligned_raster_window(
        env,
        partitions,
        spatial_chunk=args.spatial_chunk,
    )
    shard_window_seconds = time.perf_counter() - phase
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}

    plan_rows = [item.rows for item in plan]
    ideal_rows = sum(plan_rows) / len(plan_rows)
    max_rows = max(plan_rows)
    min_rows = min(plan_rows)

    local_root = Path(os.environ.get("TMPDIR", output_dir)) / (
        f"hrhsa-point-balanced-{os.environ.get('SLURM_JOB_ID', 'local')}-{args.shard_index}"
    )
    local_root.mkdir(parents=True, exist_ok=True)
    memory_limit_bytes = int(args.managed_memory_gib * 1024**3 / args.workers)

    phase = time.perf_counter()
    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
        memory_limit=memory_limit_bytes,
        local_directory=str(local_root),
        dashboard_address=":0",
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
        cluster_startup_seconds = time.perf_counter() - phase
        rows = sum(int(part.rows or 0) for part in partitions)
        print(
            f"balanced shard {args.shard_index}/{args.shard_count}: "
            f"partitions={len(partitions)} rows={rows:,} "
            f"plan_rows_min={min_rows:,} plan_rows_max={max_rows:,} "
            f"ideal_rows={ideal_rows:,.1f} bounds={shard.bounds} "
            f"raster={int(shard_env.sizes['y'])}x{int(shard_env.sizes['x'])} "
            f"physical_cores={len(task_core_groups)} logical_cpus={len(task_cpus)}",
            flush=True,
        )
        print("worker affinity:", json.dumps(worker_affinity, sort_keys=True), flush=True)

        try:
            client.get_task_stream()
        except Exception:
            pass

        phase = time.perf_counter()
        _validate_production_path(
            partitions=partitions,
            env=shard_env,
            client=client,
            chunks=chunks,
            validation_points=min(args.validation_points, int(partitions[0].rows or 0)),
        )
        validation_seconds = time.perf_counter() - phase
        setup_to_ready_seconds = time.perf_counter() - process_started

        setup_payload = {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "point_count_full": int(args.point_count),
            "target_raster_gib_full": float(args.raster_gib),
            "input_load_seconds": input_load_seconds,
            "raster_window_selection_seconds": raster_window_selection_seconds,
            "xy_cache_seconds": xy_cache_seconds,
            "xy_cache_present_before": xy_cache_present_before,
            "xy_cache_directory": str(cache_dir),
            "shard_plan_seconds": shard_plan_seconds,
            "shard_window_seconds": shard_window_seconds,
            "cluster_startup_seconds": cluster_startup_seconds,
            "validation_seconds": validation_seconds,
            "setup_to_ready_seconds": setup_to_ready_seconds,
        }
        setup_output.write_text(
            json.dumps(setup_payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            "setup timings: "
            f"xy_cache={xy_cache_seconds:.3f}s "
            f"cache_reused={xy_cache_present_before} "
            f"cluster={cluster_startup_seconds:.3f}s "
            f"validation={validation_seconds:.3f}s "
            f"ready={setup_to_ready_seconds:.3f}s",
            flush=True,
        )

        barrier_started = time.perf_counter()
        _wait_for_barrier(
            args.barrier_dir.expanduser().resolve(),
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            timeout=args.barrier_timeout,
        )
        barrier_wait_seconds = time.perf_counter() - barrier_started
        setup_payload["barrier_wait_seconds"] = barrier_wait_seconds
        setup_output.write_text(
            json.dumps(setup_payload, indent=2, default=str) + "\n",
            encoding="utf-8",
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
                    "backend": "slurm-localcluster-balanced-shard",
                    "architecture": "production_partitioned_block_local_spatial_balanced_shard",
                    "shard_planner": "recursive_weighted_spatial_bisection",
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "partition_indices": selected_indices,
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
                    "plan_rows_min": int(min_rows),
                    "plan_rows_max": int(max_rows),
                    "plan_rows_ideal": float(ideal_rows),
                    "plan_max_over_ideal": float(max_rows / ideal_rows),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                }
            )
            _append_jsonl(output, record)
            print(
                f"balanced_shard={args.shard_index} repeat={repeat} "
                f"rows={record['rows']:,} pipeline={record['pipeline_seconds']:.3f}s "
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
