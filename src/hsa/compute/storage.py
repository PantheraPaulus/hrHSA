"""Storage-layout helpers for HPC raster stores."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr


def _resolved_chunk(env: xr.DataArray, chunks: dict[str, int], dim: str) -> int:
    value = int(chunks.get(dim, env.sizes[dim]))
    if value == -1:
        return int(env.sizes[dim])
    if value <= 0:
        raise ValueError(f"Chunk length for {dim!r} must be positive or -1")
    return min(value, int(env.sizes[dim]))


def plan_zarr_shards(
    env: xr.DataArray,
    chunks: dict[str, int],
    *,
    target_shard_mb: int = 512,
) -> dict[str, int]:
    """Plan Zarr-v3 shard dimensions as multiples of logical chunks.

    Sharding reduces the number of physical storage objects while preserving the
    smaller independently readable/compressible chunk grid.  The heuristic grows
    x/y symmetrically until the approximate uncompressed shard reaches the target.
    It is especially useful as a benchmark candidate on metadata-sensitive shared
    filesystems; it is not enabled implicitly.
    """
    if target_shard_mb <= 0:
        raise ValueError("target_shard_mb must be positive")
    if "x" not in env.dims or "y" not in env.dims:
        raise ValueError("plan_zarr_shards expects x and y dimensions")

    resolved = {
        dim: _resolved_chunk(env, chunks, dim)
        for dim in env.dims
    }
    chunk_elements = math.prod(resolved.values())
    chunk_bytes = chunk_elements * np.dtype(env.dtype).itemsize
    target_bytes = int(target_shard_mb) * 1024**2
    chunks_per_shard = max(1, target_bytes // max(1, chunk_bytes))
    spatial_factor = max(1, int(math.floor(math.sqrt(chunks_per_shard))))

    shards = dict(resolved)
    for dim in ("x", "y"):
        shards[dim] = min(
            int(env.sizes[dim]),
            resolved[dim] * spatial_factor,
        )
    if "band" in shards:
        shards["band"] = int(env.sizes["band"])
    return shards


def zarr_storage_units(env: xr.DataArray, layout: dict[str, int]) -> int:
    """Estimate physical chunk/shard objects needed for a regular layout."""
    units = 1
    for dim in env.dims:
        size = int(env.sizes[dim])
        step = int(layout.get(dim, size))
        if step == -1:
            step = size
        if step <= 0:
            raise ValueError("layout lengths must be positive or -1")
        units *= int(math.ceil(size / step))
    return int(units)
