from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def estimate_raster_bytes(
    env: xr.DataArray,
    *,
    dtype: str | np.dtype | None = None,
) -> int:
    """Estimate the dense in-memory size of an xarray raster stack in bytes."""

    itemsize = np.dtype(dtype or env.dtype).itemsize
    n_cells = 1
    for dim in env.dims:
        n_cells *= int(env.sizes[dim])
    return int(n_cells * itemsize)


def suggest_xy_chunks(
    env: xr.DataArray,
    *,
    target_chunk_mb: int = 128,
    min_xy: int = 256,
    max_xy: int = 4096,
    dtype: str | np.dtype | None = None,
) -> dict[str, int]:
    """Suggest square-ish x/y chunks for a ``band, y, x`` raster stack.

    The heuristic keeps all bands in one chunk and chooses x/y chunks so one
    block is approximately ``target_chunk_mb``. This is a sensible default for
    many RSF prediction and raster-sampling workflows, but users should still
    benchmark on real HPC storage.
    """

    if "x" not in env.dims or "y" not in env.dims:
        raise ValueError("suggest_xy_chunks expects dimensions named 'x' and 'y'.")

    n_bands = int(env.sizes.get("band", 1))
    itemsize = np.dtype(dtype or env.dtype).itemsize
    target_bytes = target_chunk_mb * 1024**2
    cells_per_xy_chunk = max(1, target_bytes // max(1, n_bands * itemsize))
    side = int(math.sqrt(cells_per_xy_chunk))
    side = max(min_xy, min(max_xy, side))

    return {
        "band": -1 if "band" in env.dims else 1,
        "y": min(side, int(env.sizes["y"])),
        "x": min(side, int(env.sizes["x"])),
    }


def rechunk_raster(
    env: xr.DataArray,
    *,
    chunks: dict[str, int] | None = None,
    target_chunk_mb: int = 128,
) -> xr.DataArray:
    """Return a Dask-chunked raster stack."""

    if chunks is None:
        chunks = suggest_xy_chunks(env, target_chunk_mb=target_chunk_mb)
    return env.chunk(chunks)


def suggest_point_batch_size(
    env: xr.DataArray,
    *,
    target_batch_mb: int = 64,
    min_points: int = 1_000,
    max_points: int = 250_000,
    safety_factor: float = 4.0,
) -> int:
    """Suggest a point batch size for raster-stack sampling.

    ``safety_factor`` accounts for dataframe overhead and intermediate arrays.
    The heuristic is intentionally conservative. Large calibrated workloads can
    override it with an explicit row batch size through :class:`ExecutionConfig`.
    """

    n_bands = int(env.sizes.get("band", 1))
    bytes_per_value = np.dtype(env.dtype).itemsize
    bytes_per_point = max(1, int(n_bands * bytes_per_value * safety_factor))
    target_bytes = target_batch_mb * 1024**2
    n_points = target_bytes // bytes_per_point
    return int(min(max_points, max(min_points, n_points)))


def iter_point_batches(
    samples: gpd.GeoDataFrame,
    *,
    batch_size: int,
) -> Iterable[gpd.GeoDataFrame]:
    """Yield bounded row slices of a GeoDataFrame without copying all rows."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(samples), batch_size):
        yield samples.iloc[start : start + batch_size]


def iter_sample_raster_stack_chunked(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray | xr.Dataset,
    *,
    batch_size: int | None = None,
    target_batch_mb: int = 64,
    batches_in_flight: int = 1,
    client=None,
    **sample_kwargs: Any,
) -> Iterator[pd.DataFrame]:
    """Yield bounded chunk-aware raster samples for a large point set.

    This is the out-of-core point-sampling primitive. Each yielded dataframe is
    produced by :func:`hsa.compute.raster.sample_raster_stack_chunked`, while the
    complete input and output never need to be materialised as one sampled table.

    Parameters
    ----------
    samples
        Point GeoDataFrame to sample.
    env
        Environmental raster stack.
    batch_size
        Explicit maximum rows per sampling call. When omitted, a conservative
        memory-based value is derived from ``target_batch_mb``.
    target_batch_mb
        Memory target used only when ``batch_size`` is omitted.
    batches_in_flight
        Maximum number of bounded sampling calls allowed to overlap. ``1`` keeps
        the established synchronous execution path. Values above one use a small
        client-side thread pool to overlap otherwise blocking sampling calls while
        preserving input/output order and a strict bound on live batch payloads.
        Higher values can increase peak memory substantially and should be chosen
        from machine/workload calibration rather than treated as a universal
        throughput knob.
    client
        Optional Dask distributed client forwarded to the chunk-aware sampler.
    **sample_kwargs
        Additional keyword arguments forwarded to
        :func:`hsa.compute.raster.sample_raster_stack_chunked`.
    """
    if not isinstance(samples, gpd.GeoDataFrame):
        raise TypeError("samples must be a geopandas.GeoDataFrame.")
    if batch_size is None:
        raster = env if isinstance(env, xr.DataArray) else env.to_array(dim="band")
        batch_size = suggest_point_batch_size(
            raster,
            target_batch_mb=target_batch_mb,
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if batches_in_flight <= 0:
        raise ValueError("batches_in_flight must be positive.")

    # Import lazily so the light-weight chunking helpers do not create an import
    # cycle while hsa.compute is initialising its public namespace.
    from hsa.compute.raster import sample_raster_stack_chunked

    batches = iter_point_batches(samples, batch_size=int(batch_size))

    # Keep the historical path free of executor overhead and, more importantly,
    # exactly synchronous unless concurrency is requested explicitly.
    if batches_in_flight == 1:
        for batch in batches:
            yield sample_raster_stack_chunked(
                batch,
                env,
                client=client,
                **sample_kwargs,
            )
        return

    executor = ThreadPoolExecutor(
        max_workers=int(batches_in_flight),
        thread_name_prefix="hrhsa-point-batch",
    )
    pending: deque[Future[pd.DataFrame]] = deque()
    try:
        for batch in batches:
            pending.append(
                executor.submit(
                    sample_raster_stack_chunked,
                    batch,
                    env,
                    client=client,
                    **sample_kwargs,
                )
            )
            if len(pending) >= batches_in_flight:
                # FIFO consumption keeps output deterministic even when later
                # sampling calls complete first.
                yield pending.popleft().result()

        while pending:
            yield pending.popleft().result()
    finally:
        # On exceptions or early generator close, do not submit replacement work.
        # Running calls are allowed to finish before the executor is torn down so
        # no client activity escapes the lifetime of this iterator.
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def sample_raster_stack_batched(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray | xr.Dataset,
    *,
    batch_size: int | None = None,
    target_batch_mb: int = 64,
    batches_in_flight: int = 1,
    client=None,
    **sample_kwargs: Any,
) -> pd.DataFrame:
    """Sample points with bounded chunk-aware calls and concatenate the result.

    Use :func:`iter_sample_raster_stack_chunked` directly when the sampled output
    itself is too large to gather in memory and should instead be consumed or
    written partition by partition.
    """

    frames = list(
        iter_sample_raster_stack_chunked(
            samples,
            env,
            batch_size=batch_size,
            target_batch_mb=target_batch_mb,
            batches_in_flight=batches_in_flight,
            client=client,
            **sample_kwargs,
        )
    )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def persist_if_dask(obj, *, client=None):
    """Persist a Dask-backed object when a client is available."""

    if not hasattr(obj, "persist"):
        return obj
    if client is None:
        return obj.persist()
    return client.persist(obj)