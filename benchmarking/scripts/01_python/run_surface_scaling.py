"""Run capacity, strong-scaling and weak-scaling surface benchmarks.

The same driver supports a local workstation and an existing CoolMUC-4 Slurm
allocation.  All problem sizes are chunk-aligned nested windows of one prepared
Zarr raster, so a campaign does not duplicate large benchmark datasets.

Definitions
-----------
capacity
    Fixed hardware and fixed calibrated geometry; increase raster size.
strong
    Fixed raster size; increase hardware while preserving worker thread width
    locally or per-node geometry on CoolMUC-4.
weak
    Increase raster size proportionally with physical cores/nodes so logical
    GiB per physical core remains approximately constant.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import psutil
import xarray as xr

from hsa.compute import (
    COOLMUC4_POLICY,
    ExecutionConfig,
    append_benchmark_record,
    coolmuc4_plan,
    current_slurm_allocation,
    discover_runtime_topology,
    slurm_allocation_client,
    spatial_task_count,
    validate_worker_topology,
)
from hsa.compute.workloads import (
    geometry_for_core_budget,
    parse_geometry,
    parse_positive_floats,
    parse_positive_ints,
    raster_workload_for_gib,
)
from run_planned_execution import _allocation_physical_cores_per_node
from run_worker_geometry import _model, _run_once


def _memory_limit(total_gib: float, workers: int, fraction: float) -> str:
    return f"{(total_gib * fraction / workers):.3f}GiB"


def _load_family(root: Path) -> tuple[xr.DataArray, dict[str, Any]]:
    manifest_path = root / "surface_family_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Prepare the nested raster family first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    env = xr.open_zarr(root / "environment.zarr", chunks={})["environment"]
    return env, manifest


def _window(full_env: xr.DataArray, target_gib: float, storage_chunk: int):
    workload = raster_workload_for_gib(
        target_gib,
        bands=int(full_env.sizes["band"]),
        dtype=full_env.dtype,
        storage_chunk=storage_chunk,
    )
    if workload.size > int(full_env.sizes["x"]) or workload.size > int(full_env.sizes["y"]):
        raise RuntimeError(
            f"Target {target_gib:g} GiB requires {workload.size}x{workload.size}, "
            f"but prepared raster is only "
            f"{full_env.sizes['x']}x{full_env.sizes['y']}."
        )
    env = full_env.isel(
        y=slice(0, workload.size),
        x=slice(0, workload.size),
    )
    return env, workload



def _run_surface_repeats(
    *,
    client,
    env: xr.DataArray,
    output: Path,
    chunks: dict[str, int],
    workers: int,
    threads_per_worker: int,
    chunk_mb: int,
    warmup_repeats: int,
    repeats: int,
    metadata: dict[str, Any],
) -> None:
    task_count = spatial_task_count(env, chunks)
    model, scaler, spec, meta = _model(env)
    benchmark_args = SimpleNamespace(
        workers=workers,
        threads_per_worker=threads_per_worker,
        chunk_mb=chunk_mb,
        warmup_repeats=warmup_repeats,
    )

    try:
        client.get_task_stream()
    except Exception:
        pass

    for warmup in range(1, warmup_repeats + 1):
        print(
            f"warm-up {warmup}/{warmup_repeats}: "
            f"{workers}x{threads_per_worker}, "
            f"{metadata['actual_raster_gib']:.3f} GiB"
        )
        _run_once(
            "surface",
            points=None,
            env=env,
            chunks=chunks,
            client=client,
            args=benchmark_args,
            task_count=task_count,
            repeat=0,
            model=model,
            scaler=scaler,
            spec=spec,
            meta=meta,
        )
        gc.collect()

    for repeat in range(1, repeats + 1):
        record = _run_once(
            "surface",
            points=None,
            env=env,
            chunks=chunks,
            client=client,
            args=benchmark_args,
            task_count=task_count,
            repeat=repeat,
            model=model,
            scaler=scaler,
            spec=spec,
            meta=meta,
        )
        record.benchmark = f"surface_{metadata['scaling_mode']}_scaling"
        record.metadata.update(
            {
                **metadata,
                "repeat": repeat,
                "warmup_repeats": warmup_repeats,
                "spatial_task_count": task_count,
                "computational_chunks": dict(chunks),
            }
        )
        append_benchmark_record(record, output)
        print(pd.Series(record.to_dict()).to_string())
        gc.collect()


def _local_configuration(
    *,
    workers: int,
    threads: int,
    total_memory_gib: float,
    managed_memory_fraction: float,
    local_directory: Path,
    chunk_mb: int,
    worker_startup_timeout: float,
) -> ExecutionConfig:
    return ExecutionConfig(
        backend="local",
        n_workers=workers,
        threads_per_worker=threads,
        processes=True,
        memory_limit=_memory_limit(
            total_memory_gib,
            workers,
            managed_memory_fraction,
        ),
        local_directory=str(local_directory),
        dashboard_address=None,
        chunk_mb=chunk_mb,
        worker_startup_timeout=worker_startup_timeout,
    )


def _base_metadata(
    *,
    args,
    workload,
    resource_kind: str,
    resource_count: int,
    physical_cores_used: int,
    geometry: str,
    backend: str,
    memory_gib: float | None,
) -> dict[str, Any]:
    return {
        "campaign": f"surface-{args.mode}-scaling-v1",
        "scaling_mode": args.mode,
        "backend": backend,
        "resource_kind": resource_kind,
        "resource_count": resource_count,
        "physical_cores_used": physical_cores_used,
        "base_geometry": args.geometry,
        "executed_geometry": geometry,
        "target_raster_gib": workload.target_gib,
        "actual_raster_gib": workload.logical_gib,
        "raster_logical_bytes": workload.logical_bytes,
        "raster_cells": workload.cells,
        "raster_size": workload.size,
        "raster_size_error_pct": workload.relative_error_fraction * 100.0,
        "logical_gib_per_physical_core": workload.logical_gib / physical_cores_used,
        "memory_gib_available_to_campaign": memory_gib,
        "raster_to_memory_ratio": (
            None
            if memory_gib in (None, 0)
            else workload.logical_gib / float(memory_gib)
        ),
        "materialization": "distributed_persist_no_final_assembly",
    }


def _campaign_points(args, *, physical_local: int | None, slurm_cores_per_node: int | None):
    base_workers, base_threads = parse_geometry(args.geometry)

    if args.backend == "local":
        if physical_local is None:
            raise RuntimeError("local physical-core count unavailable")
        if base_workers * base_threads > physical_local:
            raise ValueError(
                f"Base geometry {args.geometry} needs "
                f"{base_workers * base_threads} cores, machine has {physical_local}."
            )

        if args.mode == "capacity":
            targets = args.raster_gib
            if not targets:
                raise ValueError("--raster-gib is required for capacity mode")
            cores = base_workers * base_threads
            return [
                {
                    "target_gib": target,
                    "workers": base_workers,
                    "threads": base_threads,
                    "resource_kind": "physical_cores",
                    "resource_count": cores,
                    "physical_cores": cores,
                }
                for target in targets
            ]

        core_counts = args.core_counts
        if not core_counts:
            raise ValueError("--core-counts is required for local strong/weak scaling")
        if any(value > physical_local for value in core_counts):
            raise ValueError(
                f"Requested local core count exceeds {physical_local}: {core_counts}"
            )

        points = []
        for cores in core_counts:
            workers, threads = geometry_for_core_budget(args.geometry, cores)
            target = (
                args.fixed_raster_gib
                if args.mode == "strong"
                else cores * args.gib_per_core
            )
            if target is None:
                raise ValueError("--fixed-raster-gib is required for strong scaling")
            points.append(
                {
                    "target_gib": target,
                    "workers": workers,
                    "threads": threads,
                    "resource_kind": "physical_cores",
                    "resource_count": cores,
                    "physical_cores": cores,
                }
            )
        return points

    if slurm_cores_per_node is None:
        raise RuntimeError("Slurm physical cores/node unavailable")
    node_counts = [args.nodes] if args.mode == "capacity" else args.node_counts
    if not node_counts:
        raise ValueError("--node-counts is required for Slurm strong/weak scaling")

    points = []
    for nodes in node_counts:
        if args.mode == "capacity":
            if not args.raster_gib:
                raise ValueError("--raster-gib is required for capacity mode")
            for target in args.raster_gib:
                points.append(
                    {
                        "target_gib": target,
                        "nodes": nodes,
                        "resource_kind": "nodes",
                        "resource_count": nodes,
                        "physical_cores": nodes * slurm_cores_per_node,
                    }
                )
            break
        if args.mode == "strong":
            target = args.fixed_raster_gib
            if target is None:
                raise ValueError("--fixed-raster-gib is required for strong scaling")
        else:
            target = nodes * slurm_cores_per_node * args.gib_per_core
        points.append(
            {
                "target_gib": target,
                "nodes": nodes,
                "resource_kind": "nodes",
                "resource_count": nodes,
                "physical_cores": nodes * slurm_cores_per_node,
            }
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("capacity", "strong", "weak"), required=True)
    parser.add_argument("--backend", choices=("local", "slurm"), required=True)
    parser.add_argument(
        "--geometry",
        required=True,
        help=(
            "Calibrated geometry. Locally this is the full-machine geometry; "
            "on CoolMUC-4 it is the per-node geometry."
        ),
    )
    parser.add_argument("--raster-gib", type=parse_positive_floats, default=None)
    parser.add_argument("--fixed-raster-gib", type=float, default=None)
    parser.add_argument("--gib-per-core", type=float, default=2.0)
    parser.add_argument("--core-counts", type=parse_positive_ints, default=None)
    parser.add_argument("--node-counts", type=parse_positive_ints, default=None)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if args.repeats <= 0 or args.warmup_repeats < 0:
        parser.error("repeats must be positive and warmup-repeats non-negative")
    if args.spatial_chunk <= 0 or args.chunk_mb <= 0:
        parser.error("spatial-chunk and chunk-mb must be positive")
    if not 0 < args.managed_memory_fraction < 1:
        parser.error("managed-memory-fraction must lie in (0, 1)")
    if args.gib_per_core <= 0:
        parser.error("gib-per-core must be positive")
    if args.nodes <= 0:
        parser.error("nodes must be positive")

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

    full_env, family_manifest = _load_family(root)
    storage_chunk = int(
        family_manifest.get("materialized_raster", {}).get(
            "storage_chunk",
            args.spatial_chunk,
        )
    )
    if args.spatial_chunk != storage_chunk:
        print(
            "NOTE: computational chunk differs from stored chunk: "
            f"{args.spatial_chunk} vs {storage_chunk}"
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
            raise RuntimeError("--backend slurm must run inside an existing Slurm allocation")
        slurm_cores_per_node = _allocation_physical_cores_per_node(allocation)
        requested_nodes = [args.nodes] if args.mode == "capacity" else (args.node_counts or [])
        if requested_nodes and max(requested_nodes) > allocation.nodes:
            raise RuntimeError(
                f"Campaign requests {max(requested_nodes)} node(s), "
                f"allocation has {allocation.nodes}."
            )

    points = _campaign_points(
        args,
        physical_local=physical_local,
        slurm_cores_per_node=slurm_cores_per_node,
    )

    campaign_path = output.with_suffix(output.suffix + ".campaign.json")
    campaign = {
        "campaign": f"surface-{args.mode}-scaling-v1",
        "mode": args.mode,
        "backend": args.backend,
        "root": str(root),
        "output": str(output),
        "base_geometry": args.geometry,
        "family_manifest": family_manifest,
        "requested_points": points,
        "host": platform.node(),
        "driver_topology": discover_runtime_topology().as_dict(),
        "status": "running",
    }
    campaign_path.write_text(json.dumps(campaign, indent=2, default=str) + "\n", encoding="utf-8")

    local_tmp = root / "dask-tmp"
    local_tmp.mkdir(parents=True, exist_ok=True)
    chunks = {
        "band": -1,
        "y": args.spatial_chunk,
        "x": args.spatial_chunk,
    }

    try:
        for index, point in enumerate(points, start=1):
            env, workload = _window(
                full_env,
                point["target_gib"],
                storage_chunk,
            )
            print(
                f"\n=== {args.mode} point {index}/{len(points)} ===\n"
                f"target={workload.target_gib:g} GiB "
                f"actual={workload.logical_gib:.3f} GiB "
                f"size={workload.size}x{workload.size}"
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
                    raise RuntimeError("Local scaling benchmark requires a Dask client")
                try:
                    client.wait_for_workers(
                        workers,
                        timeout=args.worker_startup_timeout,
                    )
                    metadata = _base_metadata(
                        args=args,
                        workload=workload,
                        resource_kind=point["resource_kind"],
                        resource_count=point["resource_count"],
                        physical_cores_used=point["physical_cores"],
                        geometry=geometry,
                        backend="local",
                        memory_gib=local_memory_gib,
                    )
                    _run_surface_repeats(
                        client=client,
                        env=env,
                        output=output,
                        chunks=chunks,
                        workers=workers,
                        threads_per_worker=threads,
                        chunk_mb=args.chunk_mb,
                        warmup_repeats=args.warmup_repeats,
                        repeats=args.repeats,
                        metadata=metadata,
                    )
                finally:
                    client.close()
                    if cluster is not None:
                        cluster.close()

            else:
                nodes = int(point["nodes"])
                task_count = spatial_task_count(env, chunks)
                plan = coolmuc4_plan(
                    workload="surface_prediction",
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
                    / f"surface-scaling-{allocation.job_id}-n{nodes}-{index}.json"
                )
                with slurm_allocation_client(
                    plan,
                    scheduler_file=scheduler_file,
                    interface=args.interface,
                    worker_startup_timeout=args.worker_startup_timeout,
                    validate_topology=False,
                ) as client:
                    validation = validate_worker_topology(
                        client,
                        plan,
                        strict_affinity=True,
                    )
                    print(validation.explain())
                    if not validation.ok:
                        raise RuntimeError(
                            "Strict physical-core topology validation failed"
                        )
                    memory_gib = (
                        None
                        if allocation.memory_per_node_gib is None
                        else allocation.memory_per_node_gib * nodes
                    )
                    metadata = {
                        **_base_metadata(
                            args=args,
                            workload=workload,
                            resource_kind=point["resource_kind"],
                            resource_count=point["resource_count"],
                            physical_cores_used=point["physical_cores"],
                            geometry=plan.geometry.label,
                            backend="slurm",
                            memory_gib=memory_gib,
                        ),
                        "execution_plan": plan.as_dict(),
                        "topology_validation": validation.as_dict(),
                        "slurm_job_id": allocation.job_id,
                        "slurm_partition": allocation.partition,
                        "slurm_node_list": allocation.node_list,
                        "allocation_nodes": allocation.nodes,
                        "node_subset": nodes,
                    }
                    _run_surface_repeats(
                        client=client,
                        env=env,
                        output=output,
                        chunks=chunks,
                        workers=plan.total_workers,
                        threads_per_worker=plan.geometry.threads_per_worker,
                        chunk_mb=args.chunk_mb,
                        warmup_repeats=args.warmup_repeats,
                        repeats=args.repeats,
                        metadata=metadata,
                    )

        campaign["status"] = "completed"
    except BaseException as exc:
        campaign["status"] = "failed"
        campaign["error"] = repr(exc)
        raise
    finally:
        campaign["completed_records_file"] = str(output)
        campaign_path.write_text(
            json.dumps(campaign, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
