"""Chunk-aware raster extraction kernels for large point samples.

The reference sampler in :mod:`hsa.sampling` uses labelled xarray vectorised
indexing, which is ideal for ordinary analyses. At HPC scale the expensive part
is often storage access rather than nearest-neighbour arithmetic. The functions
below route points to raster chunks first, load each touched chunk once, and then
perform NumPy gathering inside the chunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hsa.compute.planning import plan_raster_chunks
from hsa.sampling import _as_raster_dataarray


def _nearest_indices(coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return nearest indices in a monotonic one-dimensional coordinate array.

    Regular raster coordinates use an arithmetic fast path. Irregular monotonic
    coordinates retain the general ``searchsorted`` implementation. Midpoint ties
    follow xarray/pandas nearest-index semantics and resolve toward the larger
    coordinate value.
    """
    coordinates = np.asarray(coordinates)
    values = np.asarray(values)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("coordinates must be a non-empty one-dimensional array.")
    if coordinates.size == 1:
        return np.zeros(values.shape, dtype=np.int64)

    descending = bool(coordinates[0] > coordinates[-1])
    if descending:
        reversed_index = _nearest_indices(coordinates[::-1], values)
        return (coordinates.size - 1 - reversed_index).astype(np.int64, copy=False)

    delta = np.diff(coordinates)
    if np.any(delta <= 0):
        raise ValueError("Raster x/y coordinates must be strictly monotonic.")

    # Most geospatial rasters have affine/regular coordinates. In that case the
    # nearest cell can be derived directly instead of running two binary searches
    # for every point. The one-ULP nudge preserves the established midpoint rule.
    step = float(delta[0])
    scale = max(1.0, float(np.max(np.abs(coordinates))))
    tolerance = max(
        abs(step) * 1e-12,
        np.finfo(np.float64).eps * scale * 8.0,
    )
    regular = bool(np.all(np.abs(delta - step) <= tolerance))
    if regular:
        fractional = (values - float(coordinates[0])) / step
        shifted = np.nextafter(fractional + 0.5, np.inf)
        indices = np.floor(shifted).astype(np.int64, copy=False)
        return np.clip(indices, 0, coordinates.size - 1)

    right = np.searchsorted(coordinates, values, side="left")
    right = np.clip(right, 1, coordinates.size - 1)
    left = right - 1
    choose_right = np.abs(values - coordinates[right]) <= np.abs(values - coordinates[left])
    return np.where(choose_right, right, left).astype(np.int64, copy=False)


def _chunk_lengths(size: int, nominal: int) -> tuple[int, ...]:
    """Expand a nominal chunk size into concrete lengths including the tail."""
    if nominal <= 0:
        raise ValueError("nominal chunk length must be positive.")
    full, remainder = divmod(int(size), int(nominal))
    lengths = [int(nominal)] * full
    if remainder:
        lengths.append(int(remainder))
    return tuple(lengths or [int(size)])


def _dimension_chunk_lengths(
    raster: xr.DataArray,
    dimension: str,
    *,
    nominal: int,
) -> tuple[int, ...]:
    """Return actual Dask chunks when present, otherwise use ``nominal``."""
    data = raster.data
    chunks = getattr(data, "chunks", None)
    if chunks is None:
        return _chunk_lengths(int(raster.sizes[dimension]), nominal)
    axis = raster.get_axis_num(dimension)
    return tuple(int(value) for value in chunks[axis])


def _chunk_number(index: np.ndarray, lengths: tuple[int, ...]) -> np.ndarray:
    """Map valid absolute indices to zero-based chunk numbers.

    Ordinary Dask raster chunks have one nominal width plus an optional shorter
    tail. For that common layout, integer division avoids a binary search for
    every point. Irregular chunk layouts retain the general ``searchsorted``
    fallback.
    """
    lengths_array = np.asarray(lengths, dtype=np.int64)
    if lengths_array.ndim != 1 or lengths_array.size == 0:
        raise ValueError("lengths must be a non-empty one-dimensional sequence.")
    if np.any(lengths_array <= 0):
        raise ValueError("chunk lengths must be positive.")

    index = np.asarray(index, dtype=np.int64)
    if lengths_array.size == 1:
        return np.zeros(index.shape, dtype=np.int64)

    nominal = int(lengths_array[0])
    regular = bool(
        np.all(lengths_array[:-1] == nominal)
        and int(lengths_array[-1]) <= nominal
    )
    if regular:
        return (index // nominal).astype(np.int64, copy=False)

    boundaries = np.cumsum(lengths_array)
    return np.searchsorted(boundaries, index, side="right").astype(np.int64)


def _chunk_starts(lengths: tuple[int, ...]) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(np.asarray(lengths[:-1], dtype=np.int64))))


def _group_positions_by_chunk(
    chunk_y: np.ndarray,
    chunk_x: np.ndarray,
    *,
    n_x_chunks: int,
) -> list[tuple[int, int, np.ndarray]]:
    """Group point positions by spatial chunk using vectorized integer sorting.

    The previous implementation appended every point to a Python ``defaultdict``.
    That work is necessarily serial and becomes measurable for multi-million-point
    samples. Encoding each (chunk_y, chunk_x) pair as one integer lets NumPy perform
    the expensive grouping work in compiled code; Python then loops only over the
    comparatively small number of touched chunks.
    """
    chunk_y = np.asarray(chunk_y, dtype=np.int64)
    chunk_x = np.asarray(chunk_x, dtype=np.int64)
    if chunk_y.shape != chunk_x.shape:
        raise ValueError("chunk_y and chunk_x must have matching shapes.")
    if n_x_chunks <= 0:
        raise ValueError("n_x_chunks must be positive.")
    if chunk_y.size == 0:
        return []

    encoded = chunk_y * int(n_x_chunks) + chunk_x
    order = np.argsort(encoded, kind="stable")
    sorted_ids = encoded[order]
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]).astype(np.int64) + 1,
        )
    )
    stops = np.concatenate((starts[1:], np.array([order.size], dtype=np.int64)))

    grouped: list[tuple[int, int, np.ndarray]] = []
    for start, stop in zip(starts, stops):
        chunk_id = int(sorted_ids[int(start)])
        cy, cx = divmod(chunk_id, int(n_x_chunks))
        grouped.append((cy, cx, order[int(start) : int(stop)]))
    return grouped


def _extract_numpy_block(block: Any, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Gather all bands for point-local row/column indices from one block."""
    array = np.asarray(block)
    if array.ndim != 3:
        raise ValueError("Raster block must have dimensions (band, y, x).")
    return np.asarray(array[:, rows, cols])


def _extract_numpy_block_flat(
    block: Any,
    flat_indices: np.ndarray,
) -> np.ndarray:
    """Gather all bands using one packed within-block point-index payload."""
    array = np.asarray(block)
    if array.ndim != 3:
        raise ValueError("Raster block must have dimensions (band, y, x).")

    flat_indices = np.asarray(flat_indices)
    if flat_indices.ndim != 1:
        raise ValueError("flat_indices must be one-dimensional.")

    flat = array.reshape(array.shape[0], -1)
    return np.asarray(flat[:, flat_indices])


def _normalise_id_cols(id_cols: str | Sequence[str] | None) -> list[str]:
    if id_cols is None:
        return []
    if isinstance(id_cols, str):
        return [id_cols]
    return list(id_cols)


def sample_raster_stack_chunked(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray | xr.Dataset,
    bands: Sequence[str] | None = None,
    *,
    dtype: str | np.dtype = "float32",
    id_cols: str | Sequence[str] | None = None,
    chunks: dict[str, int] | None = None,
    target_chunk_mb: int = 256,
    align_storage: bool = True,
    client=None,
    require_inside: bool = False,
) -> pd.DataFrame:
    """Sample a raster by routing points to spatial chunks before extraction.

    This function is numerically equivalent in intent to
    :func:`hsa.sampling.sample_raster_stack` for nearest-neighbour extraction, but
    changes the execution order to reduce repeated storage reads. Every spatial
    chunk touched by the point set is loaded once per call and all point values
    inside that chunk are gathered together.

    Parameters
    ----------
    samples
        Point GeoDataFrame.
    env
        Environmental stack with dimensions ``band, y, x``.
    bands
        Optional subset of bands. Selecting bands before chunking avoids loading
        unused variables from broad Earth-observation stacks.
    chunks
        Explicit computational chunks. If omitted, hrHSA's workload-aware planner
        uses the memory target and, by default, aligns to source storage chunks.
        The selected band dimension is kept in one chunk.
    client
        Optional Dask distributed client. When supplied, touched chunk tasks are
        computed through that scheduler. Point-local indices are flattened within
        each raster chunk and packed into one unsigned integer payload before they
        are scattered, reducing scheduler/communication overhead for large point
        batches. Without a client, Dask's local scheduler is used when the raster
        is Dask-backed; NumPy rasters remain purely local.
    require_inside
        If True, reject points outside the raster extent rather than snapping
        them to the nearest edge cell.
    """
    if not isinstance(samples, gpd.GeoDataFrame):
        raise TypeError("samples must be a geopandas.GeoDataFrame.")
    if samples.crs is None:
        raise ValueError("samples.crs is None; set a CRS before raster sampling.")

    raster = _as_raster_dataarray(env)
    if bands is not None:
        raster = raster.sel(band=list(bands))
    raster = raster.transpose("band", "y", "x")

    try:
        env_crs = raster.rio.crs
    except Exception:
        env_crs = None

    transformed = samples
    if env_crs is not None and samples.crs != env_crs:
        transformed = samples.to_crs(env_crs)

    x_coordinates = np.asarray(raster["x"].values)
    y_coordinates = np.asarray(raster["y"].values)
    point_x = transformed.geometry.x.to_numpy(dtype=float)
    point_y = transformed.geometry.y.to_numpy(dtype=float)

    if require_inside and len(samples):
        outside = (
            (point_x < float(np.min(x_coordinates)))
            | (point_x > float(np.max(x_coordinates)))
            | (point_y < float(np.min(y_coordinates)))
            | (point_y > float(np.max(y_coordinates)))
        )
        if np.any(outside):
            raise ValueError(
                f"{int(outside.sum()):,} points fall outside the environmental raster extent."
            )

    row_index = _nearest_indices(y_coordinates, point_y)
    col_index = _nearest_indices(x_coordinates, point_x)

    if chunks is None:
        chunks = plan_raster_chunks(
            raster,
            workload="point_sampling",
            target_chunk_mb=target_chunk_mb,
            align_storage=align_storage,
            selected_band_count=int(raster.sizes["band"]),
        )
    chunks = dict(chunks)
    chunks["band"] = -1

    # Rechunk only when Dask is available/already backing the object. Keeping the
    # NumPy path free of an optional Dask dependency is useful for reference tests.
    data_chunks = getattr(raster.data, "chunks", None)
    if data_chunks is not None:
        raster = raster.chunk(chunks)

    y_nominal = int(chunks.get("y", raster.sizes["y"]))
    x_nominal = int(chunks.get("x", raster.sizes["x"]))
    y_lengths = _dimension_chunk_lengths(raster, "y", nominal=y_nominal)
    x_lengths = _dimension_chunk_lengths(raster, "x", nominal=x_nominal)
    y_starts = _chunk_starts(y_lengths)
    x_starts = _chunk_starts(x_lengths)

    chunk_y = _chunk_number(row_index, y_lengths)
    chunk_x = _chunk_number(col_index, x_lengths)
    grouped_positions = _group_positions_by_chunk(
        chunk_y,
        chunk_x,
        n_x_chunks=len(x_lengths),
    )

    n_bands = int(raster.sizes["band"])
    values = np.empty((n_bands, len(samples)), dtype=np.dtype(dtype))

    delayed_tasks = []
    delayed_positions: list[np.ndarray] = []
    distributed_specs: list[tuple[Any, np.ndarray]] = []
    dask_backed = getattr(raster.data, "chunks", None) is not None

    for cy, cx, positions in grouped_positions:
        y_start = int(y_starts[cy])
        x_start = int(x_starts[cx])
        y_stop = y_start + int(y_lengths[cy])
        x_stop = x_start + int(x_lengths[cx])
        block_width = int(x_lengths[cx])

        local_rows = row_index[positions] - y_start
        local_cols = col_index[positions] - x_start
        max_flat_index = int(y_lengths[cy]) * block_width - 1
        index_dtype = (
            np.uint32
            if max_flat_index <= np.iinfo(np.uint32).max
            else np.uint64
        )
        flat_indices = (
            local_rows * block_width + local_cols
        ).astype(index_dtype, copy=False)
        block = raster.isel(y=slice(y_start, y_stop), x=slice(x_start, x_stop)).data

        if dask_backed:
            try:
                import dask
            except ImportError as exc:  # pragma: no cover - broken optional env
                raise ImportError(
                    "A Dask-backed raster requires the optional hsa[dask] or hsa[hpc] dependencies."
                ) from exc
            if client is None:
                delayed_tasks.append(
                    dask.delayed(_extract_numpy_block_flat)(block, flat_indices)
                )
            else:
                # Keep large point-index payloads out of the Dask task graph. The
                # flattened unsigned payloads are scattered below and represented
                # by small Future keys while block loading and extraction stay fused.
                distributed_specs.append((block, flat_indices))
            delayed_positions.append(positions)
        else:
            values[:, positions] = _extract_numpy_block_flat(
                block,
                flat_indices,
            ).astype(dtype, copy=False)

    if delayed_tasks:
        import dask

        computed = dask.compute(*delayed_tasks)
        for positions, extracted in zip(delayed_positions, computed):
            values[:, positions] = np.asarray(extracted, dtype=dtype)

    elif distributed_specs:
        import dask

        index_payloads = [flat_indices for _, flat_indices in distributed_specs]
        index_futures = client.scatter(
            index_payloads,
            broadcast=False,
            hash=False,
        )
        extraction_futures = []
        try:
            distributed_tasks = [
                dask.delayed(_extract_numpy_block_flat)(block, index_future)
                for (block, _), index_future in zip(distributed_specs, index_futures)
            ]
            extraction_futures = client.compute(distributed_tasks)
            computed = client.gather(extraction_futures)
            for positions, extracted in zip(delayed_positions, computed):
                values[:, positions] = np.asarray(extracted, dtype=dtype)
        finally:
            futures_to_cancel = [*index_futures, *extraction_futures]
            if futures_to_cancel:
                client.cancel(futures_to_cancel)

    sampled_bands = [str(value) for value in raster["band"].values]
    out = pd.DataFrame(values.T, columns=sampled_bands, index=samples.index)
    out["x"] = point_x
    out["y"] = point_y

    for column in ("used", "Timestamp"):
        if column in samples.columns:
            out[column] = samples[column].to_numpy()

    for column in _normalise_id_cols(id_cols):
        if column not in samples.columns:
            raise ValueError(f"id_cols not found in samples.columns: {column!r}")
        out[column] = samples[column].to_numpy()

    return out


def compare_sampling_engines(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray | xr.Dataset,
    *,
    bands: Sequence[str] | None = None,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Run reference and chunk-aware samplers and report numerical agreement.

    This helper is intended for development notebooks and benchmark smoke tests.
    It is not a performance benchmark because it intentionally computes both paths.
    """
    from hsa.sampling import sample_raster_stack

    reference = sample_raster_stack(samples, env, bands=bands)
    accelerated = sample_raster_stack_chunked(samples, env, bands=bands)
    raster = _as_raster_dataarray(env)
    if bands is not None:
        raster = raster.sel(band=list(bands))
    value_columns = [str(value) for value in raster.band.values]
    ref_values = reference[value_columns].to_numpy(dtype=float)
    accelerated_values = accelerated[value_columns].to_numpy(dtype=float)
    difference = np.abs(ref_values - accelerated_values)
    max_abs = float(np.nanmax(difference)) if difference.size else 0.0
    return {
        "equivalent": bool(
            np.allclose(ref_values, accelerated_values, atol=atol, equal_nan=True)
        ),
        "max_abs_difference": max_abs,
        "n_points": int(len(samples)),
        "n_bands": int(len(value_columns)),
    }