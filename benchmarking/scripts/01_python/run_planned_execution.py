"""Run raster benchmarks through the public topology-aware HPC execution path."""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import xarray as xr

from hsa.compute import (
    COOLMUC4_HARDWARE,
    COOLMUC4_POLICY,
    append_benchmark_record,
    build_run_manifest,
    coolmuc4_plan,
    current_slurm_allocation,
    finalize_run_manifest,
    slurm_allocation_client,
    spatial_task_count,
    validate_worker_topology,
    write_run_manifest,
)
from run_worker_geometry import _model, _parse_names, _run_once


def _allocation_physical_cores_per_node(allocation) -> int:
    """Resolve CoolMUC physical cores from Slurm scheduler CPU units.

    On ``cm4_inter`` a full 112-physical-core allocation can expose
    ``SLURM_JOB_CPUS_PER_NODE=224`` because Slurm reports both SMT hardware
    threads. hrHSA plans are intentionally expressed in physical cores. Prefer
    the allocation's explicit task decomposition when present and otherwise map
    a full logical-CPU report back to the known 112-core node topology.
    """
    if allocation.tasks_per_node:
        requested = allocation.tasks_per_node * (allocation.cpus_per_task or 1)
        if requested > 0:
            return requested

    reported = allocation.cpus_per_node
    if reported is None:
        return COOLMUC4_HARDWARE.physical_cores_per_node

    if (
        reported > COOLMUC4_HARDWARE.physical_cores_per_node
        and reported <= COOLMUC4_HARDWARE.logical_cores_per_node
    ):
        return COOLMUC4_HARDWARE.physical_cores_per_node

    return reported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmarks", type=_parse_names, default=["surface"])
    parser.add_argument("--geometry", default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--interface", default=None)
    parser.add_argument(
        "--campaign",
        default="coolmuc4-planned-execution",
        help="Campaign label written to JSONL records and the run manifest.",
    )
    parser.add_argument(
        "--strict-affinity",
        action="store_true",
        help=(
            "Fail unless Dask worker placement matches the requested physical-core "
            "geometry. SMT sibling CPU ids may remain visible under core binding."
        ),
    )
    args = parser.parse_args()

    if args.repeats <= 0 or args.warmup_repeats < 0:
        raise ValueError("repeats must be positive and warmup-repeats non-negative")

    allocation = current_slurm_allocation()
    if allocation is None:
        raise RuntimeError("run_planned_execution.py must run inside Slurm")

    args.root = args.root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")

    env = xr.open_zarr(args.root / "environment.zarr", chunks={})["environment"]
    points = (
        gpd.read_parquet(args.root / "points.parquet")
        if "sampling" in args.benchmarks
        else None
    )
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}
    task_count = spatial_task_count(env, chunks)

    cores_per_node = _allocation_physical_cores_per_node(allocation)
    if allocation.cpus_per_node not in (None, cores_per_node):
        print(
            "Slurm CPU-unit normalization: "
            f"reported_cpus_per_node={allocation.cpus_per_node}, "
            f"physical_plan_cores_per_node={cores_per_node}, "
            f"tasks_per_node={allocation.tasks_per_node}, "
            f"cpus_per_task={allocation.cpus_per_task}"
        )

    workload = "point_sampling" if args.benchmarks == ["sampling"] else "surface_prediction"
    plan = coolmuc4_plan(
        workload=workload,
        nodes=allocation.nodes,
        cores_per_node=cores_per_node,
        geometry=args.geometry,
        task_count=task_count,
        chunk_mb=args.chunk_mb,
    )
    if allocation.partition == COOLMUC4_POLICY.interactive_partition:
        # The geometry/memory recommendation is still the CoolMUC-4 production
        # plan, but this diagnostic is physically executing in cm4_inter. Record
        # that fact in the execution plan as well as the Slurm allocation record.
        plan = replace(plan, partition=allocation.partition)
    elif allocation.partition and allocation.partition != plan.partition:
        raise RuntimeError(
            "Current Slurm partition does not match the CoolMUC-4 execution plan: "
            f"allocation={allocation.partition!r}, plan={plan.partition!r}"
        )
    print(plan.explain())

    model, scaler, spec, meta = _model(env)
    benchmark_args = SimpleNamespace(
        workers=plan.total_workers,
        threads_per_worker=plan.geometry.threads_per_worker,
        chunk_mb=plan.chunk_mb,
        warmup_repeats=args.warmup_repeats,
    )

    scheduler_file = args.output.parent / f"planned-scheduler-{allocation.job_id}.json"
    manifest = None
    try:
        # Keep launcher validation disabled here only so an invalid physical
        # placement can still be captured in a failure manifest before timing is
        # refused. Production callers keep the launcher's validation enabled.
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
                strict_affinity=args.strict_affinity,
            )
            print(validation.explain())

            dataset_metadata = {
                "root": str(args.root),
                "environment_shape": list(env.shape),
                "environment_dims": list(env.dims),
                "environment_dtype": str(env.dtype),
                "computational_chunks": chunks,
                "spatial_task_count": task_count,
                "sampling_rows": None if points is None else len(points),
            }
            manifest = build_run_manifest(
                plan,
                client=client,
                dataset=dataset_metadata,
                extra={
                    "campaign": args.campaign,
                    "strict_affinity": args.strict_affinity,
                    "output_jsonl": str(args.output),
                    "slurm_reported_cpus_per_node": allocation.cpus_per_node,
                    "physical_plan_cores_per_node": cores_per_node,
                },
                strict_affinity=args.strict_affinity,
            )
            write_run_manifest(manifest, manifest_path)
            print("Run manifest:", manifest_path)

            if not validation.ok:
                raise RuntimeError(
                    "Observed Dask/Slurm topology does not match the execution plan; "
                    f"diagnostics were written to {manifest_path}"
                )

            for warmup in range(1, args.warmup_repeats + 1):
                for benchmark in args.benchmarks:
                    print(
                        f"Warm-up {warmup}/{args.warmup_repeats}: benchmark={benchmark}, "
                        f"geometry={plan.geometry.label}, nodes={plan.nodes}"
                    )
                    _run_once(
                        benchmark,
                        points=points,
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

            for repeat in range(1, args.repeats + 1):
                for benchmark in args.benchmarks:
                    record = _run_once(
                        benchmark,
                        points=points,
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
                    record.metadata.update(
                        {
                            "campaign": args.campaign,
                            "execution_plan": plan.as_dict(),
                            "nodes": plan.nodes,
                            "manifest": str(manifest_path),
                            "topology_validation": validation.as_dict(),
                            "slurm_reported_cpus_per_node": allocation.cpus_per_node,
                            "physical_plan_cores_per_node": cores_per_node,
                        }
                    )
                    append_benchmark_record(record, args.output)
                    print(pd.Series(record.to_dict()).to_string())
                    gc.collect()

        if manifest is not None:
            finalize_run_manifest(manifest, manifest_path, status="completed")
    except BaseException:
        if manifest is not None:
            finalize_run_manifest(manifest, manifest_path, status="failed")
        raise


if __name__ == "__main__":
    main()
