"""Benchmark the production partition-native point sampler inside a Slurm allocation.

This is the CoolMUC-4 companion to ``profile_point_partitioned_production.py``.
It reuses the same x/y cache, validation and telemetry helpers, but launches the
Dask scheduler/workers through hrHSA's allocation-aware ``coolmuc4_plan()`` and
``slurm_allocation_client()`` path.  It is intentionally one workload point per
invocation so an enclosing sbatch script can build capacity matrices while
retaining one JSONL output.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from pathlib import Path

from hsa.compute import (
    COOLMUC4_POLICY,
    coolmuc4_plan,
    current_slurm_allocation,
    slurm_allocation_client,
    spatial_task_count,
    validate_worker_topology,
)
from hsa.compute.workloads import parse_geometry
from profile_point_partitioned_production import (
    _append_jsonl,
    _materialize_xy_cache,
    _run_once,
    _validate_production_path,
)
from run_planned_execution import _allocation_physical_cores_per_node
from run_point_scaling import _load_inputs, _point_workload_entry
from run_surface_scaling import _window


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark production partition-native point sampling in Slurm."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, required=True)
    parser.add_argument("--raster-gib", type=float, required=True)
    parser.add_argument("--geometry", default="8x14")
    parser.add_argument("--graph-partitions", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--interface", default=None)
    args = parser.parse_args()

    if min(
        args.point_count,
        args.graph_partitions,
        args.nodes,
        args.spatial_chunk,
        args.chunk_mb,
        args.validation_points,
        args.repeats,
    ) <= 0:
        parser.error("point/graph/node/chunk/validation/repeat values must be positive")
    if args.raster_gib <= 0:
        parser.error("raster-gib must be positive")

    allocation = current_slurm_allocation()
    if allocation is None:
        raise RuntimeError("This runner must execute inside an existing Slurm allocation.")
    if args.nodes > allocation.nodes:
        raise RuntimeError(
            f"Requested {args.nodes} nodes but allocation contains {allocation.nodes}."
        )

    cores_per_node = _allocation_physical_cores_per_node(allocation)
    requested_workers, requested_threads = parse_geometry(args.geometry)
    if requested_workers * requested_threads != int(cores_per_node):
        raise ValueError(
            f"Geometry {args.geometry} uses {requested_workers * requested_threads} "
            f"physical cores/node; allocation exposes {cores_per_node}. "
            "Use a full-node geometry for the frozen capacity campaign."
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

    # Synthetic u/v -> real x/y preparation is deliberately outside benchmark
    # timers.  Subsequent campaign runs reuse this shared-scratch cache.
    partitions = _materialize_xy_cache(root, entry, env)

    task_count = spatial_task_count(env, chunks)
    plan = coolmuc4_plan(
        workload="point_sampling",
        nodes=args.nodes,
        cores_per_node=int(cores_per_node),
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

    scheduler_file = output.parent / (
        f"point-partitioned-{allocation.job_id}-"
        f"{args.point_count}p-{int(round(raster_workload.logical_gib))}g.json"
    )

    with slurm_allocation_client(
        plan,
        scheduler_file=scheduler_file,
        interface=args.interface,
        worker_startup_timeout=args.worker_startup_timeout,
        validate_topology=False,
    ) as client:
        validation = validate_worker_topology(client, plan, strict_affinity=True)
        print(validation.explain())
        if not validation.ok:
            raise RuntimeError("Strict physical-core topology validation failed.")

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
                workers=plan.total_workers,
                threads_per_worker=plan.geometry.threads_per_worker,
            )
            record.update(
                {
                    "repeat": repeat,
                    "backend": "slurm",
                    "geometry": plan.geometry.label,
                    "workers": plan.total_workers,
                    "threads_per_worker": plan.geometry.threads_per_worker,
                    "execution_threads": (
                        plan.total_workers * plan.geometry.threads_per_worker
                    ),
                    "nodes": args.nodes,
                    "physical_cores_per_node": int(cores_per_node),
                    "point_count": int(args.point_count),
                    "target_raster_gib": float(args.raster_gib),
                    "raster_gib": raster_workload.logical_gib,
                    "spatial_chunk": args.spatial_chunk,
                    "architecture": "production_partitioned_block_local",
                    "execution_plan": plan.as_dict(),
                    "topology_validation": validation.as_dict(),
                    "slurm_job_id": allocation.job_id,
                    "slurm_partition": allocation.partition,
                    "slurm_node_list": allocation.node_list,
                }
            )
            _append_jsonl(output, record)
            print(
                f"repeat {repeat}: points={args.point_count:,} "
                f"raster={raster_workload.logical_gib:.3f}GiB "
                f"pipeline={record['pipeline_seconds']:.3f}s "
                f"points/s={record['throughput_points_s']:,.0f} "
                f"worker_cores={record['worker_busy_cores_pipeline']} "
                f"task_parallelism={record['task_compute_parallelism']} "
                f"tail={record['pipeline_minus_task_span_seconds']} "
                f"rss_mb={record['operation_peak_process_tree_rss_mb']}"
            )
            gc.collect()


if __name__ == "__main__":
    main()
