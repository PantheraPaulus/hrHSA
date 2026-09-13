"""Prepare one reusable, storage-backed hrHSA benchmark dataset.

The generated raster is written chunk-wise to Zarr so preparation does not build
an enormous in-memory NumPy array. Points are stored once as GeoParquet and reused
across every worker-count run. The resulting files are intended for a fast shared
scratch filesystem such as CoolMUC-4 ``$SCRATCH_DSS``.

The default 32768 x 32768 x 6 float32 raster is about 24 GiB uncompressed and has
1024 spatial chunks at 1024 x 1024. That deliberately provides about nine chunks
per worker on a 112-core CoolMUC-4 node; use ``--size 65536`` (about 96 GiB,
4096 chunks) for the 1/2/4-node scaling campaign.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import dask
import dask.array as da
import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401 - register .rio
import xarray as xr


def _make_raster(
    path: Path,
    *,
    size: int,
    bands: int,
    storage_chunk: int,
    seed: int,
    workers: int,
) -> None:
    rng = da.random.default_rng(seed)
    data = rng.normal(
        size=(bands, size, size),
        chunks=(bands, storage_chunk, storage_chunk),
    ).astype("float32")
    x = np.arange(size, dtype="float64") * 30.0
    y = np.arange(size - 1, -1, -1, dtype="float64") * 30.0
    env = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={
            "band": [f"x{index}" for index in range(bands)],
            "y": y,
            "x": x,
        },
        name="environment",
    ).rio.write_crs("EPSG:3857")
    with dask.config.set(scheduler="threads", num_workers=workers):
        env.to_dataset().to_zarr(path, mode="w")


def _make_points(path: Path, *, size: int, n_points: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    maximum = float((size - 1) * 30.0)
    x = rng.uniform(0.0, maximum, n_points)
    y = rng.uniform(0.0, maximum, n_points)
    points = gpd.GeoDataFrame(
        {"used": rng.random(n_points) < 0.1},
        geometry=gpd.points_from_xy(x, y),
        crs="EPSG:3857",
    )
    points.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=32_768)
    parser.add_argument("--bands", type=int, default=6)
    parser.add_argument("--storage-chunk", type=int, default=1024)
    parser.add_argument("--points", type=int, default=5_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    raster_path = root / "environment.zarr"
    points_path = root / "points.parquet"
    manifest_path = root / "manifest.json"

    if args.force:
        if raster_path.exists():
            shutil.rmtree(raster_path)
        if points_path.exists():
            points_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    if not raster_path.exists():
        _make_raster(
            raster_path,
            size=args.size,
            bands=args.bands,
            storage_chunk=args.storage_chunk,
            seed=args.seed,
            workers=args.workers,
        )
    if not points_path.exists():
        _make_points(
            points_path,
            size=args.size,
            n_points=args.points,
            seed=args.seed + 1,
        )

    raster_bytes = int(args.bands * args.size * args.size * np.dtype("float32").itemsize)
    manifest = {
        "size": args.size,
        "bands": args.bands,
        "storage_chunk": args.storage_chunk,
        "points": args.points,
        "seed": args.seed,
        "preparation_workers": args.workers,
        "raster_uncompressed_bytes": raster_bytes,
        "raster_uncompressed_gib": raster_bytes / 1024**3,
        "spatial_chunks": int(np.ceil(args.size / args.storage_chunk) ** 2),
        "raster": str(raster_path),
        "points_file": str(points_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
