"""Reproducible synthetic benchmark harness for hrHSA HPC kernels.

Examples
--------
Serial reference/accelerated comparison::

    python benchmarks/benchmark_hpc.py --benchmark sampling --workers 1

Strong-scaling point for four local workers::

    python benchmarks/benchmark_hpc.py --benchmark sampling --workers 4 --output results.jsonl

Large distributed storage benchmarks should use ``benchmarks/hpc/run_scaling.py``
against a pre-generated Zarr store rather than increasing the in-memory raster.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers the xarray .rio accessor
import xarray as xr

from hsa import FeatureSpec
from hsa.compute import (
    ExecutionConfig,
    append_benchmark_record,
    make_benchmark_record,
    sample_raster_stack_chunked,
    spatial_task_count,
    wall_timer,
)
from hsa.compute.persistence import cancel_distributed, persist_distributed
from hsa.rsf import fit_rsf, predict_rsf_surface_chunked


def synthetic_raster(size: int, bands: int, *, seed: int) -> xr.DataArray:
    """Create a deterministic in-memory raster for local microbenchmarks."""
    rng = np.random.default_rng(seed)
    x = np.arange(size, dtype=float) * 30.0
    y = np.arange(size - 1, -1, -1, dtype=float) * 30.0
    data = rng.normal(size=(bands, size, size)).astype("float32")
    env = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={
            "band": [f"x{index}" for index in range(bands)],
            "y": y,
            "x": x,
        },
    )
    return env.rio.write_crs("EPSG:3857")


def synthetic_points(env: xr.DataArray, n: int, *, seed: int) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    xmin, xmax = float(env.x.min()), float(env.x.max())
    ymin, ymax = float(env.y.min()), float(env.y.max())
    x = rng.uniform(xmin, xmax, n)
    y = rng.uniform(ymin, ymax, n)
    return gpd.GeoDataFrame(
        {"used": rng.random(n) < 0.1},
        geometry=gpd.points_from_xy(x, y),
        crs="EPSG:3857",
    )


def sampling_benchmark(
    env: xr.DataArray,
    points: gpd.GeoDataFrame,
    *,
    client,
    workers: int,
    chunk_mb: int,
):
    chunks = {
        "band": -1,
        "y": min(2048, env.sizes["y"]),
        "x": min(2048, env.sizes["x"]),
    }
    if client is not None:
        env = env.chunk(chunks)
    tasks = spatial_task_count(env, chunks)
    with wall_timer() as timer:
        sampled = sample_raster_stack_chunked(
            points,
            env,
            chunks=chunks,
            client=client,
        )
    assert len(sampled) == len(points)
    return make_benchmark_record(
        "chunked_point_sampling",
        timer["wall_seconds"],
        rows=len(points),
        workers=workers,
        threads_per_worker=1,
        chunk_mb=chunk_mb,
        tasks=tasks,
        metadata={
            "bands": int(env.sizes["band"]),
            "raster_size": int(env.sizes["x"]),
            "source": "synthetic-memory",
        },
        client=client,
    )


def surface_benchmark(
    env: xr.DataArray,
    points: gpd.GeoDataFrame,
    *,
    client,
    workers: int,
    chunk_mb: int,
):
    sampled = sample_raster_stack_chunked(points, env)
    spec = FeatureSpec(
        linear=list(map(str, env.band.values)),
        quadratic=[str(env.band.values[0])],
        interactions=[(str(env.band.values[0]), str(env.band.values[1]))]
        if env.sizes["band"] > 1
        else [],
    )
    model, scaler, fitted_spec, meta = fit_rsf(sampled, spec)
    chunks = {
        "band": -1,
        "y": min(2048, env.sizes["y"]),
        "x": min(2048, env.sizes["x"]),
    }
    raster = env.chunk(chunks) if client is not None else env
    tasks = spatial_task_count(env, chunks)
    persisted = None

    try:
        with wall_timer() as timer:
            predicted = predict_rsf_surface_chunked(
                raster,
                model,
                scaler,
                fitted_spec,
                meta,
                chunks=chunks,
            )
            if hasattr(predicted.data, "compute"):
                if client is None:
                    predicted.compute()
                else:
                    persisted = persist_distributed(client, predicted.data)
            else:
                np.asarray(predicted.values)

        cells = int(env.sizes["x"] * env.sizes["y"])
        record = make_benchmark_record(
            "fused_surface_prediction",
            timer["wall_seconds"],
            rows=cells,
            workers=workers,
            threads_per_worker=1,
            chunk_mb=chunk_mb,
            tasks=tasks,
            metadata={
                "bands": int(env.sizes["band"]),
                "raster_size": int(env.sizes["x"]),
                "source": "synthetic-memory",
                "materialization": (
                    "distributed_persist_no_final_assembly"
                    if client is not None
                    else "local_compute"
                ),
                "output_partitions": (
                    int(getattr(persisted, "npartitions", 1))
                    if persisted is not None
                    else None
                ),
            },
            client=client,
        )
        return record
    finally:
        cancel_distributed(client, persisted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["sampling", "surface"],
        default="sampling",
    )
    parser.add_argument("--raster-size", type=int, default=4096)
    parser.add_argument("--bands", type=int, default=6)
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results.jsonl"),
    )
    args = parser.parse_args()

    env = synthetic_raster(args.raster_size, args.bands, seed=args.seed)
    points = synthetic_points(env, args.points, seed=args.seed + 1)
    client = None
    if args.workers > 1:
        execution = ExecutionConfig(
            backend="local",
            n_workers=args.workers,
            threads_per_worker=1,
            chunk_mb=args.chunk_mb,
            dashboard_address=None,
        )
        client, _ = execution.create_client()

    try:
        if args.benchmark == "sampling":
            record = sampling_benchmark(
                env,
                points,
                client=client,
                workers=args.workers,
                chunk_mb=args.chunk_mb,
            )
        else:
            record = surface_benchmark(
                env,
                points,
                client=client,
                workers=args.workers,
                chunk_mb=args.chunk_mb,
            )
        append_benchmark_record(record, args.output)
        print(pd.Series(record.to_dict()).to_string())
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
