from __future__ import annotations

from pathlib import Path

import xarray as xr

from hsa.compute.chunking import rechunk_raster, suggest_xy_chunks
from hsa.compute.storage import plan_zarr_shards


def write_raster_stack_zarr(
    env: xr.DataArray,
    path: str | Path,
    *,
    name: str = "env",
    mode: str = "w",
    chunks: dict[str, int] | None = None,
    target_chunk_mb: int = 128,
    shards: dict[str, int] | None = None,
    target_shard_mb: int | None = None,
    zarr_format: int | None = None,
    consolidated: bool = True,
    compute: bool = True,
):
    """Write a raster stack to Zarr using HPC-friendly chunks.

    Parameters
    ----------
    shards, target_shard_mb
        Optional Zarr-v3 sharding. Shards group multiple logical chunks into fewer
        physical storage objects, which can reduce metadata pressure on shared HPC
        filesystems while retaining chunk-level reads. Use either explicit
        ``shards`` or ``target_shard_mb``. Sharding is opt-in because the optimum
        depends on the storage system and access pattern.
    zarr_format
        Explicit Zarr format. Supplying sharding requires format 3; otherwise the
        installed Xarray/Zarr defaults are preserved.

    Notes
    -----
    For sharded stores, the logical Zarr chunk grid remains ``chunks`` while the
    Dask write graph is rechunked to complete ``shards``. Current Xarray validates
    parallel writes at the shard boundary, so writing complete shards avoids
    multiple Dask tasks mutating the same physical shard and preserves its safe
    chunk-alignment guarantees.

    Returns
    -------
    object
        The object returned by :meth:`xarray.Dataset.to_zarr`. With
        ``compute=False`` this can be a delayed object suitable for explicit Dask
        execution.
    """
    if chunks is None:
        chunks = suggest_xy_chunks(env, target_chunk_mb=target_chunk_mb)
    chunks = dict(chunks)

    if shards is not None and target_shard_mb is not None:
        raise ValueError("Use either shards or target_shard_mb, not both.")

    # Start from the intended logical chunk grid. This is also the input used by
    # the shard planner when only a target shard size is supplied.
    env = rechunk_raster(env, chunks=chunks)
    if target_shard_mb is not None:
        shards = plan_zarr_shards(
            env,
            chunks,
            target_shard_mb=target_shard_mb,
        )

    encoding = None
    if shards is not None:
        if zarr_format not in {None, 3}:
            raise ValueError("Zarr sharding requires zarr_format=3.")
        zarr_format = 3
        shard_tuple = tuple(
            int(env.sizes[dim]) if int(shards.get(dim, env.sizes[dim])) == -1
            else int(shards.get(dim, env.sizes[dim]))
            for dim in env.dims
        )
        chunk_tuple = tuple(
            int(env.sizes[dim]) if int(chunks.get(dim, env.sizes[dim])) == -1
            else int(chunks.get(dim, env.sizes[dim]))
            for dim in env.dims
        )
        for chunk_size, shard_size, dim in zip(chunk_tuple, shard_tuple, env.dims):
            if shard_size < chunk_size or shard_size % chunk_size != 0:
                raise ValueError(
                    f"Shard length for {dim!r} must be a multiple of the chunk "
                    f"length ({shard_size} vs {chunk_size})."
                )

        # Xarray/Zarr can safely write a sharded array in parallel when each Dask
        # task owns whole shards. Keep the smaller logical Zarr chunks in encoding,
        # but make the Dask write grid equal to the shard grid.
        shard_chunks = {
            dim: int(shards.get(dim, env.sizes[dim]))
            for dim in env.dims
        }
        env = rechunk_raster(env, chunks=shard_chunks)
        encoding = {name: {"chunks": chunk_tuple, "shards": shard_tuple}}

    ds = env.to_dataset(name=name)
    kwargs = {
        "mode": mode,
        "consolidated": consolidated,
        "compute": compute,
    }
    if encoding is not None:
        kwargs["encoding"] = encoding
    if zarr_format is not None:
        kwargs["zarr_format"] = zarr_format
    return ds.to_zarr(path, **kwargs)


def open_raster_stack_zarr(
    path: str | Path,
    *,
    name: str = "env",
    chunks: dict[str, int] | str | None = "auto",
    consolidated: bool | None = None,
) -> xr.DataArray:
    """Open a raster stack from Zarr."""

    ds = xr.open_zarr(path, chunks=chunks, consolidated=consolidated)
    if name not in ds:
        raise KeyError(
            f"Variable {name!r} not found in Zarr store. "
            f"Available variables: {list(ds.data_vars)}"
        )
    return ds[name]


def write_table_parquet(df, path: str | Path, *, index: bool = False, **kwargs) -> None:
    """Write a dataframe to Parquet.

    This helper keeps tabular intermediate output consistent across examples.
    It requires either ``pyarrow`` or ``fastparquet`` in the environment.
    """

    df.to_parquet(path, index=index, **kwargs)


def read_table_parquet(path: str | Path, **kwargs):
    """Read a dataframe from Parquet."""

    import pandas as pd

    return pd.read_parquet(path, **kwargs)
