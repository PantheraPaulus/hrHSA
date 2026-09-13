"""Benchmark Dask process/thread geometry at fixed physical-core concurrency."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hsa import FeatureSpec
from hsa.compute import (
    append_benchmark_record,
    benchmark_timer,
    make_benchmark_record,
    sample_raster_stack_chunked,
    spatial_task_count,
)
from hsa.compute.persistence import persist_distributed, release_distributed
from hsa.rsf import predict_rsf_surface_chunked


def _parse_names(value: str) -> list[str]:
    allowed = {"sampling", "surface"}
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or any(item not in allowed for item in result):
        raise argparse.ArgumentTypeError("benchmarks must be sampling and/or surface")
    return result


def _model(env: xr.DataArray):
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


def _metadata(args, env, task_count: int, repeat: int) -> dict:
    concurrency = args.workers * args.threads_per_worker
    return {
        "repeat": repeat,
        "warmup_repeats": args.warmup_repeats,
        "campaign": "coolmuc4-worker-geometry",
        "geometry": f"{args.workers}x{args.threads_per_worker}",
        "worker_processes": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "total_worker_threads": concurrency,
        "tasks_per_execution_slot": task_count / concurrency,
        "raster_size_x": int(env.sizes["x"]),
        "raster_size_y": int(env.sizes["y"]),
        "bands": int(env.sizes["band"]),
        "materialization": "distributed_persist_no_final_assembly",
        "memory_monitor_interval_seconds": 0.25,
    }


def _sampling_once(
    *,
    points,
    env,
    chunks,
    client,
    args,
    task_count: int,
    repeat: int,
):
    with benchmark_timer(client=client) as timer:
        sampled = sample_raster_stack_chunked(
            points,
            env,
            chunks=chunks,
            client=client,
        )
    if len(sampled) != len(points):
        raise RuntimeError("Sampling benchmark returned the wrong row count")
    record = make_benchmark_record(
        "coolmuc4_geometry_point_sampling",
        timer["wall_seconds"],
        rows=len(points),
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        chunk_mb=args.chunk_mb,
        tasks=task_count,
        bytes_processed=int(env.nbytes),
        metadata=_metadata(args, env, task_count, repeat),
        operation_memory=timer,
        client=client,
    )
    del sampled
    gc.collect()
    return record


def _surface_once(
    *,
    env,
    chunks,
    client,
    args,
    task_count: int,
    repeat: int,
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

        record = make_benchmark_record(
            "coolmuc4_geometry_surface_prediction",
            timer["wall_seconds"],
            rows=int(env.sizes["x"] * env.sizes["y"]),
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            chunk_mb=args.chunk_mb,
            tasks=task_count,
            bytes_processed=int(env.nbytes),
            metadata={
                **_metadata(args, env, task_count, repeat),
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


def _run_once(
    benchmark: str,
    *,
    points,
    env,
    chunks,
    client,
    args,
    task_count: int,
    repeat: int,
    model,
    scaler,
    spec,
    meta,
):
    if benchmark == "sampling":
        if points is None:
            raise RuntimeError("Sampling benchmark requires points")
        return _sampling_once(
            points=points,
            env=env,
            chunks=chunks,
            client=client,
            args=args,
            task_count=task_count,
            repeat=repeat,
        )

    return _surface_once(
        env=env,
        chunks=chunks,
        client=client,
        args=args,
        task_count=task_count,
        repeat=repeat,
        model=model,
        scaler=scaler,
        spec=spec,
        meta=meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--threads-per-worker", type=int, required=True)
    parser.add_argument("--benchmarks", type=_parse_names, default=["sampling", "surface"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.workers <= 0 or args.threads_per_worker <= 0 or args.repeats <= 0:
        raise ValueError("workers, threads-per-worker and repeats must be positive")
    if args.warmup_repeats < 0:
        raise ValueError("warmup-repeats cannot be negative")

    args.root = args.root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = xr.open_zarr(args.root / "environment.zarr", chunks={})["environment"]
    points = (
        gpd.read_parquet(args.root / "points.parquet")
        if "sampling" in args.benchmarks
        else None
    )
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}
    task_count = spatial_task_count(env, chunks)
    model, scaler, spec, meta = _model(env)

    from dask.distributed import Client

    client = Client(scheduler_file=str(args.scheduler_file))
    try:
        client.wait_for_workers(args.workers, timeout=args.worker_startup_timeout)
        info = client.scheduler_info().get("workers", {})
        if len(info) != args.workers:
            raise RuntimeError(
                f"Expected {args.workers} Dask workers, observed {len(info)}"
            )
        nthreads = {int(item.get("nthreads", 0)) for item in info.values()}
        if nthreads != {args.threads_per_worker}:
            raise RuntimeError(
                f"Expected {args.threads_per_worker} threads/worker; observed {sorted(nthreads)}"
            )

        for warmup in range(1, args.warmup_repeats + 1):
            for benchmark in args.benchmarks:
                print(
                    f"Warm-up {warmup}/{args.warmup_repeats}: "
                    f"benchmark={benchmark}, geometry="
                    f"{args.workers}x{args.threads_per_worker}"
                )
                _run_once(
                    benchmark,
                    points=points,
                    env=env,
                    chunks=chunks,
                    client=client,
                    args=args,
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
                    args=args,
                    task_count=task_count,
                    repeat=repeat,
                    model=model,
                    scaler=scaler,
                    spec=spec,
                    meta=meta,
                )
                append_benchmark_record(record, args.output)
                print(pd.Series(record.to_dict()).to_string())
    finally:
        client.close()


if __name__ == "__main__":
    main()
