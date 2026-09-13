"""Run reproducible strong-scaling benchmarks against one prepared Zarr store."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hsa import FeatureSpec
from hsa.compute import (
    ExecutionConfig,
    append_benchmark_record,
    benchmark_timer,
    make_benchmark_record,
    recommend_worker_count,
    sample_raster_stack_chunked,
    spatial_task_count,
    task_density,
)
from hsa.compute.persistence import persist_distributed, release_distributed
from hsa.rsf import predict_rsf_surface_chunked


def _parse_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("worker counts must be positive integers")
    return result


def _parse_names(value: str) -> list[str]:
    allowed = {"sampling", "surface"}
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or any(item not in allowed for item in result):
        raise argparse.ArgumentTypeError("benchmarks must be sampling and/or surface")
    return result


def _open_env(root: Path) -> xr.DataArray:
    return xr.open_zarr(root / "environment.zarr", chunks={})["environment"]


def _deterministic_model(env: xr.DataArray):
    bands = [str(value) for value in env.band.values]
    spec = FeatureSpec(
        linear=bands,
        quadratic=[bands[0]],
        interactions=[(bands[0], bands[1])] if len(bands) > 1 else [],
        add_const=True,
    )
    params: dict[str, float] = {"const": -0.5}
    for index, band in enumerate(bands):
        params[band] = 0.08 * (1 if index % 2 == 0 else -1)
    params[f"{bands[0]}__sq"] = 0.02
    if len(bands) > 1:
        params[f"{bands[0]}__x__{bands[1]}"] = -0.015
    model = SimpleNamespace(params=pd.Series(params))
    scaler = SimpleNamespace(
        mean_=np.zeros(len(bands), dtype=float),
        scale_=np.ones(len(bands), dtype=float),
    )
    meta = {"categorical": {}, "columns": list(params)}
    return model, scaler, spec, meta


def _execution(args, workers: int) -> ExecutionConfig:
    if args.backend == "local":
        return ExecutionConfig(
            backend="local",
            n_workers=workers,
            threads_per_worker=1,
            memory_limit=args.local_worker_memory,
            local_directory=os.environ.get("TMPDIR", "dask-tmp"),
            dashboard_address=None,
            chunk_mb=args.chunk_mb,
            worker_startup_timeout=args.worker_startup_timeout,
        )
    if args.backend == "scheduler-file":
        raise RuntimeError("scheduler-file backend uses an externally created client")

    if args.processes_per_job <= 0:
        raise ValueError("--processes-per-job must be positive for SLURM")
    prologue = [
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
    ]
    if args.worker_prologue:
        prologue.extend(args.worker_prologue)

    directives = list(args.job_extra_directive or [])
    return ExecutionConfig(
        backend="slurm",
        n_workers=workers,
        threads_per_worker=1,
        local_directory="$TMPDIR",
        dashboard_address=None,
        chunk_mb=args.chunk_mb,
        worker_startup_timeout=args.worker_startup_timeout,
        slurm_options={
            "queue": args.partition,
            "project": args.project,
            "cores": args.cores_per_job,
            "processes": args.processes_per_job,
            "memory": args.memory_per_job,
            "walltime": args.worker_walltime,
            "job_extra_directives": directives,
            "env_extra": prologue,
            "scheduler_options": {"dashboard_address": None},
        },
    )


def _record_metadata(args, *, repeat: int, task_count: int, workers: int, env: xr.DataArray) -> dict:
    return {
        "repeat": repeat,
        "warmup_repeats": args.warmup_repeats,
        "backend": args.backend,
        "data_root": str(args.root),
        "raster_size_x": int(env.sizes["x"]),
        "raster_size_y": int(env.sizes["y"]),
        "bands": int(env.sizes["band"]),
        "task_density": task_density(task_count, workers),
        "bytes_basis": "uncompressed_raster_input",
        "memory_monitor_interval_seconds": 0.25,
        "local_worker_memory": args.local_worker_memory if args.backend == "local" else None,
    }


def _sampling_once(
    env: xr.DataArray,
    points: gpd.GeoDataFrame,
    *,
    chunks: dict[str, int],
    client,
    workers: int,
    args,
    repeat: int,
    task_count: int,
):
    with benchmark_timer(client=client) as timer:
        sampled = sample_raster_stack_chunked(
            points,
            env,
            chunks=chunks,
            client=client,
        )
    if len(sampled) != len(points):
        raise RuntimeError("Sampling benchmark returned the wrong row count.")
    record = make_benchmark_record(
        "hpc_zarr_point_sampling",
        timer["wall_seconds"],
        rows=len(points),
        workers=workers,
        threads_per_worker=1,
        chunk_mb=args.chunk_mb,
        tasks=task_count,
        bytes_processed=int(env.nbytes),
        metadata=_record_metadata(
            args,
            repeat=repeat,
            task_count=task_count,
            workers=workers,
            env=env,
        ),
        operation_memory=timer,
        client=client,
    )
    del sampled
    gc.collect()
    return record


def _surface_once(
    env: xr.DataArray,
    *,
    chunks: dict[str, int],
    client,
    workers: int,
    args,
    repeat: int,
    task_count: int,
    model,
    scaler,
    spec,
    meta,
):
    persisted = None
    predicted = None
    try:
        with benchmark_timer(client=client) as timer:
            predicted = predict_rsf_surface_chunked(
                env,
                model,
                scaler,
                spec,
                meta,
                chunks=chunks,
            )
            persisted = persist_distributed(client, predicted.data)

        cells = int(env.sizes["x"] * env.sizes["y"])
        record = make_benchmark_record(
            "hpc_zarr_surface_prediction",
            timer["wall_seconds"],
            rows=cells,
            workers=workers,
            threads_per_worker=1,
            chunk_mb=args.chunk_mb,
            tasks=task_count,
            bytes_processed=int(env.nbytes),
            metadata={
                **_record_metadata(
                    args,
                    repeat=repeat,
                    task_count=task_count,
                    workers=workers,
                    env=env,
                ),
                "materialization": "distributed_persist_no_final_assembly",
                "output_partitions": int(getattr(persisted, "npartitions", 1)),
            },
            operation_memory=timer,
            client=client,
        )
        return record
    finally:
        release_distributed(client, persisted)
        del persisted, predicted
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=_parse_ints, required=True)
    parser.add_argument("--benchmarks", type=_parse_names, default=["sampling", "surface"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--min-tasks-per-worker", type=int, default=8)
    parser.add_argument("--allow-low-task-density", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["local", "slurm", "scheduler-file"],
        default="local",
    )
    parser.add_argument("--scheduler-file", type=Path, default=None)
    parser.add_argument("--local-worker-memory", default="auto")
    parser.add_argument("--worker-startup-timeout", type=float, default=1800.0)

    parser.add_argument("--partition", default="cm4_std")
    parser.add_argument("--project", default=None)
    parser.add_argument("--cores-per-job", type=int, default=112)
    parser.add_argument("--processes-per-job", type=int, default=112)
    parser.add_argument("--memory-per-job", default="480GB")
    parser.add_argument("--worker-walltime", default="00:45:00")
    parser.add_argument("--job-extra-directive", action="append", default=[])
    parser.add_argument("--worker-prologue", action="append", default=[])
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmup_repeats < 0:
        raise ValueError("--warmup-repeats cannot be negative")
    if args.spatial_chunk <= 0:
        raise ValueError("--spatial-chunk must be positive")
    if args.backend == "scheduler-file":
        if args.scheduler_file is None:
            raise ValueError("--scheduler-file is required for scheduler-file backend")
        if len(args.workers) != 1:
            raise ValueError("scheduler-file backend requires exactly one worker count")

    args.root = args.root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = _open_env(args.root)
    points = (
        gpd.read_parquet(args.root / "points.parquet")
        if "sampling" in args.benchmarks
        else None
    )
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}
    task_count = spatial_task_count(env, chunks)
    model, scaler, spec, meta = _deterministic_model(env)

    for workers in args.workers:
        recommended = recommend_worker_count(
            task_count,
            workers,
            min_tasks_per_worker=args.min_tasks_per_worker,
        )
        if recommended < workers and not args.allow_low_task_density:
            print(
                f"Skipping {workers} workers: {task_count} spatial tasks give "
                f"only {task_density(task_count, workers):.2f} tasks/worker; "
                f"heuristic recommendation is <= {recommended}."
            )
            continue

        cluster = None
        if args.backend == "scheduler-file":
            from dask.distributed import Client

            client = Client(scheduler_file=str(args.scheduler_file))
        else:
            execution = _execution(args, workers)
            client, cluster = execution.create_client()
        if client is None:
            raise RuntimeError("HPC scaling benchmark requires a Dask client.")
        try:
            client.wait_for_workers(workers, timeout=args.worker_startup_timeout)
            observed_workers = len(client.scheduler_info().get("workers", {}))
            if observed_workers != workers:
                raise RuntimeError(
                    f"Expected {workers} workers but scheduler reports {observed_workers}."
                )
            print(
                f"Running workers={workers}, tasks={task_count}, "
                f"tasks/worker={task_density(task_count, workers):.2f}"
            )
            for benchmark in args.benchmarks:
                for warmup in range(1, args.warmup_repeats + 1):
                    print(
                        f"Warm-up {warmup}/{args.warmup_repeats}: "
                        f"benchmark={benchmark}, workers={workers}"
                    )
                    if benchmark == "sampling":
                        if points is None:
                            raise RuntimeError("Sampling benchmark requires points.")
                        _sampling_once(
                            env,
                            points,
                            chunks=chunks,
                            client=client,
                            workers=workers,
                            args=args,
                            repeat=0,
                            task_count=task_count,
                        )
                    else:
                        _surface_once(
                            env,
                            chunks=chunks,
                            client=client,
                            workers=workers,
                            args=args,
                            repeat=0,
                            task_count=task_count,
                            model=model,
                            scaler=scaler,
                            spec=spec,
                            meta=meta,
                        )

                for repeat in range(1, args.repeats + 1):
                    if benchmark == "sampling":
                        if points is None:
                            raise RuntimeError("Sampling benchmark requires points.")
                        record = _sampling_once(
                            env,
                            points,
                            chunks=chunks,
                            client=client,
                            workers=workers,
                            args=args,
                            repeat=repeat,
                            task_count=task_count,
                        )
                    else:
                        record = _surface_once(
                            env,
                            chunks=chunks,
                            client=client,
                            workers=workers,
                            args=args,
                            repeat=repeat,
                            task_count=task_count,
                            model=model,
                            scaler=scaler,
                            spec=spec,
                            meta=meta,
                        )
                    append_benchmark_record(record, args.output)
                    print(pd.Series(record.to_dict()).to_string())
        finally:
            client.close()
            if cluster is not None:
                cluster.close()


if __name__ == "__main__":
    main()
