"""Zarr-backed scaling smoke benchmark for GitHub Actions.

This benchmark writes the synthetic environmental stack to a chunked Zarr store
first and then benchmarks all worker counts against the same lazily opened store.
That mirrors the intended HPC execution model and avoids embedding a large
in-memory NumPy array in every Dask graph.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401 - registers the xarray .rio accessor
import xarray as xr

from hsa import FeatureSpec
from hsa.compute import (
    ExecutionConfig,
    make_benchmark_record,
    sample_raster_stack_chunked,
    spatial_task_count,
    wall_timer,
)
from hsa.compute.persistence import cancel_distributed, persist_distributed
from hsa.rsf import fit_rsf, predict_rsf_surface_chunked


RASTER_SIZE = 3072
N_BANDS = 4
N_POINTS = 150_000
SPATIAL_CHUNK = 768
WORKERS = (1, 2, 4)
REPEATS = 2
CHUNK_MB = 64
SEED = 42


def make_store(path: Path) -> None:
    """Create one deterministic chunked environmental Zarr store."""
    rng = np.random.default_rng(SEED)
    x = np.arange(RASTER_SIZE, dtype=float) * 30.0
    y = np.arange(RASTER_SIZE - 1, -1, -1, dtype=float) * 30.0
    data = rng.normal(size=(N_BANDS, RASTER_SIZE, RASTER_SIZE)).astype("float32")
    env = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={
            "band": [f"x{index}" for index in range(N_BANDS)],
            "y": y,
            "x": x,
        },
        name="environment",
    ).rio.write_crs("EPSG:3857")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    env.to_dataset().chunk(
        {"band": -1, "y": SPATIAL_CHUNK, "x": SPATIAL_CHUNK}
    ).to_zarr(path, mode="w")


def open_store(path: Path) -> xr.DataArray:
    ds = xr.open_zarr(path, chunks={})
    return ds["environment"].chunk(
        {"band": -1, "y": SPATIAL_CHUNK, "x": SPATIAL_CHUNK}
    )


def make_points(env: xr.DataArray) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(SEED + 1)
    xmin, xmax = float(env.x.min()), float(env.x.max())
    ymin, ymax = float(env.y.min()), float(env.y.max())
    xx = rng.uniform(xmin, xmax, N_POINTS)
    yy = rng.uniform(ymin, ymax, N_POINTS)
    return gpd.GeoDataFrame(
        {"used": rng.random(N_POINTS) < 0.1},
        geometry=gpd.points_from_xy(xx, yy),
        crs="EPSG:3857",
    )


def append_record(path: Path, record, *, repeat: int) -> None:
    payload = record.to_dict()
    payload.setdefault("metadata", {})["repeat"] = repeat
    payload["metadata"]["source"] = "synthetic-zarr"
    payload["metadata"]["storage_chunk"] = SPATIAL_CHUNK
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def main() -> None:
    # CI outputs live under the current benchmarking tree. Keep transient data in
    # one dedicated folder so directory reorganisations cannot silently leave the
    # workflow pointing at the retired ``benchmarks/`` path.
    run_dir = Path("benchmarking/ci")
    store = run_dir / "_ci_environment.zarr"
    output = run_dir / "ci-zarr-results.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    make_store(store)
    env = open_store(store)
    points = make_points(env)
    chunks = {"band": -1, "y": SPATIAL_CHUNK, "x": SPATIAL_CHUNK}
    tasks = spatial_task_count(env, chunks)

    # Fit once outside the timed surface benchmark.
    fit_samples = sample_raster_stack_chunked(points, env, chunks=chunks)
    spec = FeatureSpec(
        linear=list(map(str, env.band.values)),
        quadratic=[str(env.band.values[0])],
        interactions=[(str(env.band.values[0]), str(env.band.values[1]))],
    )
    model, scaler, fitted_spec, meta = fit_rsf(fit_samples, spec)

    for workers in WORKERS:
        execution = ExecutionConfig(
            backend="local",
            n_workers=workers,
            threads_per_worker=1,
            chunk_mb=CHUNK_MB,
            dashboard_address=None,
        )
        client, cluster = execution.create_client()
        try:
            client.wait_for_workers(workers)

            for repeat in range(1, REPEATS + 1):
                with wall_timer() as timer:
                    sampled = sample_raster_stack_chunked(
                        points,
                        env,
                        chunks=chunks,
                        client=client,
                    )
                assert len(sampled) == len(points)
                sampling = make_benchmark_record(
                    "zarr_point_sampling",
                    timer["wall_seconds"],
                    rows=len(points),
                    workers=workers,
                    threads_per_worker=1,
                    chunk_mb=CHUNK_MB,
                    tasks=tasks,
                    bytes_processed=int(env.nbytes),
                    metadata={"bands": N_BANDS, "raster_size": RASTER_SIZE},
                    client=client,
                )
                append_record(output, sampling, repeat=repeat)

                persisted = None
                predicted = None
                try:
                    with wall_timer() as timer:
                        predicted = predict_rsf_surface_chunked(
                            env,
                            model,
                            scaler,
                            fitted_spec,
                            meta,
                            chunks=chunks,
                        )
                        persisted = persist_distributed(client, predicted.data)
                    cells = int(env.sizes["x"] * env.sizes["y"])
                    surface = make_benchmark_record(
                        "zarr_surface_prediction",
                        timer["wall_seconds"],
                        rows=cells,
                        workers=workers,
                        threads_per_worker=1,
                        chunk_mb=CHUNK_MB,
                        tasks=tasks,
                        bytes_processed=int(env.nbytes),
                        metadata={
                            "bands": N_BANDS,
                            "raster_size": RASTER_SIZE,
                            "materialization": "distributed_persist_no_final_assembly",
                            "output_partitions": int(getattr(persisted, "npartitions", 1)),
                        },
                        client=client,
                    )
                    append_record(output, surface, repeat=repeat)
                finally:
                    cancel_distributed(client, persisted)
        finally:
            client.close()
            if cluster is not None:
                cluster.close()

    print("HRHSA_ZARR_BENCHMARK_RESULTS_BEGIN")
    print(output.read_text(encoding="utf-8"), end="")
    print("HRHSA_ZARR_BENCHMARK_RESULTS_END")


if __name__ == "__main__":
    main()
