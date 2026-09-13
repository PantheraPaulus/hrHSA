"""Compare reference and accelerated hrHSA kernels on one prepared workstation dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from hsa import FeatureSpec
from hsa.compute import sample_raster_stack_chunked
from hsa.rsf import predict_rsf_surface, predict_rsf_surface_chunked
from hsa.sampling import sample_raster_stack


def _model(env: xr.DataArray):
    bands = [str(value) for value in env.band.values]
    spec = FeatureSpec(
        linear=bands,
        quadratic=[bands[0]],
        interactions=[(bands[0], bands[1])] if len(bands) > 1 else [],
        add_const=True,
    )
    params = {"const": -0.5}
    for index, band in enumerate(bands):
        params[band] = 0.08 * (1 if index % 2 == 0 else -1)
    params[f"{bands[0]}__sq"] = 0.02
    if len(bands) > 1:
        params[f"{bands[0]}__x__{bands[1]}"] = -0.015
    return (
        SimpleNamespace(params=pd.Series(params)),
        SimpleNamespace(mean_=np.zeros(len(bands)), scale_=np.ones(len(bands))),
        spec,
        {"categorical": {}, "columns": list(params)},
    )


def _seconds(function):
    start = time.perf_counter()
    value = function()
    return time.perf_counter() - start, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Prepared workstation data directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=250_000)
    parser.add_argument("--surface-size", type=int, default=8_192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    args = parser.parse_args()

    if args.points <= 0 or args.surface_size <= 0 or args.repeats <= 0:
        raise ValueError("points, surface-size and repeats must be positive")

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = xr.open_zarr(root / "environment.zarr", chunks={})["environment"]
    points = gpd.read_parquet(root / "points.parquet")
    if args.points < len(points):
        points = points.sample(args.points, random_state=42).reset_index(drop=True)

    surface_size = min(args.surface_size, int(env.sizes["x"]), int(env.sizes["y"]))
    surface_env = env.isel(y=slice(0, surface_size), x=slice(0, surface_size))
    chunks = {
        "band": -1,
        "y": min(args.spatial_chunk, surface_size),
        "x": min(args.spatial_chunk, surface_size),
    }
    model, scaler, spec, meta = _model(surface_env)

    # Small untimed warm-up so import/graph-construction overhead and filesystem
    # metadata lookup do not dominate the algorithmic comparison.
    warm_points = points.iloc[: min(10_000, len(points))]
    with dask.config.set(scheduler="single-threaded"):
        sample_raster_stack(warm_points, env)
        sample_raster_stack_chunked(warm_points, env, chunks={"band": -1, "y": 1024, "x": 1024})
        predict_rsf_surface(surface_env.isel(y=slice(0, 1024), x=slice(0, 1024)), model, scaler, spec, meta).compute()
        predict_rsf_surface_chunked(
            surface_env.isel(y=slice(0, 1024), x=slice(0, 1024)),
            model,
            scaler,
            spec,
            meta,
            chunks={"band": -1, "y": 512, "x": 512},
        ).compute()

    rows: list[dict] = []
    reference_sample = accelerated_sample = None
    reference_surface = accelerated_surface = None

    with dask.config.set(scheduler="single-threaded"):
        for repeat in range(1, args.repeats + 1):
            # Alternate order to reduce systematic cache/order bias.
            sample_engines = ["reference", "accelerated"] if repeat % 2 else ["accelerated", "reference"]
            for engine in sample_engines:
                if engine == "reference":
                    seconds, value = _seconds(lambda: sample_raster_stack(points, env))
                    reference_sample = value
                else:
                    seconds, value = _seconds(
                        lambda: sample_raster_stack_chunked(
                            points,
                            env,
                            chunks={"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk},
                        )
                    )
                    accelerated_sample = value
                rows.append({
                    "benchmark": "point_sampling",
                    "engine": engine,
                    "repeat": repeat,
                    "wall_seconds": seconds,
                    "rows": len(points),
                })

            surface_engines = ["reference", "accelerated"] if repeat % 2 else ["accelerated", "reference"]
            for engine in surface_engines:
                if engine == "reference":
                    seconds, value = _seconds(
                        lambda: predict_rsf_surface(surface_env, model, scaler, spec, meta).compute()
                    )
                    reference_surface = value
                else:
                    seconds, value = _seconds(
                        lambda: predict_rsf_surface_chunked(
                            surface_env,
                            model,
                            scaler,
                            spec,
                            meta,
                            chunks=chunks,
                        ).compute()
                    )
                    accelerated_surface = value
                rows.append({
                    "benchmark": "surface_prediction",
                    "engine": engine,
                    "repeat": repeat,
                    "wall_seconds": seconds,
                    "rows": surface_size * surface_size,
                })

    if reference_sample is None or accelerated_sample is None:
        raise RuntimeError("Sampling comparison did not execute")
    common_columns = [str(value) for value in env.band.values]
    np.testing.assert_allclose(
        reference_sample[common_columns].to_numpy(),
        accelerated_sample[common_columns].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    if reference_surface is None or accelerated_surface is None:
        raise RuntimeError("Surface comparison did not execute")
    xr.testing.assert_allclose(reference_surface, accelerated_surface)

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "reference_vs_accelerated_raw.csv", index=False)
    summary = (
        raw.groupby(["benchmark", "engine"])
        .agg(
            repeats=("wall_seconds", "count"),
            median_seconds=("wall_seconds", "median"),
            q25_seconds=("wall_seconds", lambda x: x.quantile(0.25)),
            q75_seconds=("wall_seconds", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    reference = summary[summary.engine == "reference"][["benchmark", "median_seconds"]].rename(
        columns={"median_seconds": "reference_seconds"}
    )
    summary = summary.merge(reference, on="benchmark", how="left")
    summary["speedup_vs_reference"] = summary["reference_seconds"] / summary["median_seconds"]
    summary.to_csv(output_dir / "reference_vs_accelerated_summary.csv", index=False)

    for benchmark, subset in summary.groupby("benchmark"):
        subset = subset.set_index("engine").loc[["reference", "accelerated"]].reset_index()
        x = np.arange(len(subset))
        lower = subset["median_seconds"] - subset["q25_seconds"]
        upper = subset["q75_seconds"] - subset["median_seconds"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(x, subset["median_seconds"], yerr=np.vstack([lower, upper]), marker="o", capsize=4)
        ax.set_xticks(x, subset["engine"])
        ax.set_ylabel("Median wall time (s)")
        ax.set_title(f"{benchmark}: reference vs accelerated")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{benchmark}_reference_vs_accelerated.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
