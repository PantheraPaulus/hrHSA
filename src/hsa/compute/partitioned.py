"""Dask-native raster sampling for spatially partitioned point tables.

Large point workloads should not be materialised as one GeoDataFrame on the
scheduler/client process. Spatial Parquet tiles are read and routed on workers,
raster extraction remains block-local, and completed sampled partitions are
drained incrementally with ``as_completed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import dask
import numpy as np
import pandas as pd
import xarray as xr

from hsa.compute.planning import plan_raster_chunks
from hsa.compute.raster import (
    _chunk_number,
    _chunk_starts,
    _dimension_chunk_lengths,
    _extract_numpy_block_flat,
    _group_positions_by_chunk,
    _nearest_indices,
    _normalise_id_cols,
)
from hsa.sampling import _as_raster_dataarray

Bounds = tuple[float, float, float, float]
BlockPayload = tuple[np.ndarray, np.ndarray] | None


@dataclass(frozen=True)
class PointPartition:
    """One spatial Parquet point partition.

    ``bounds`` are ``(xmin, ymin, xmax, ymax)`` in the raster CRS. Files must be
    visible to all Dask workers. Coordinates are intentionally not reprojected
    inside this performance-sensitive primitive.
    """

    path: str | Path
    bounds: Bounds
    rows: int | None = None
    crs: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if len(self.bounds) != 4:
            raise ValueError("bounds must be (xmin, ymin, xmax, ymax).")
        xmin, ymin, xmax, ymax = (float(value) for value in self.bounds)
        if not np.all(np.isfinite([xmin, ymin, xmax, ymax])):
            raise ValueError("partition bounds must be finite.")
        if xmax < xmin or ymax < ymin:
            raise ValueError("partition bounds must satisfy xmin<=xmax and ymin<=ymax.")
        object.__setattr__(self, "bounds", (xmin, ymin, xmax, ymax))
        if self.rows is not None:
            rows = int(self.rows)
            if rows < 0:
                raise ValueError("rows must be non-negative when supplied.")
            object.__setattr__(self, "rows", rows)


@dataclass(frozen=True)
class SampledPointPartition:
    """One completed sampled partition returned in completion order."""

    partition_index: int
    source: PointPartition
    frame: pd.DataFrame


@dataclass(frozen=True)
class _RouteMeta:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    y_lengths: tuple[int, ...]
    x_lengths: tuple[int, ...]
    y_starts: np.ndarray
    x_starts: np.ndarray
    sampled_bands: tuple[str, ...]
    dtype: str
    require_inside: bool


@dataclass
class _BlockLocalRoute:
    partition_index: int
    rows: int
    point_x: np.ndarray
    point_y: np.ndarray
    preserved: tuple[np.ndarray, ...]
    payloads: tuple[BlockPayload, ...]


@dataclass(frozen=True)
class _PartitionMetadata:
    partition_index: int
    rows: int
    point_x: np.ndarray
    point_y: np.ndarray
    preserved: tuple[np.ndarray, ...]


def _normalise_columns(columns: Sequence[str] | str | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return [str(column) for column in columns]


def _deduplicate_columns(columns: Sequence[str]) -> list[str]:
    result: list[str] = []
    for column in columns:
        if column not in result:
            result.append(column)
    return result


def _same_crs(left: Any, right: Any) -> bool:
    try:
        from pyproj import CRS

        return CRS.from_user_input(left) == CRS.from_user_input(right)
    except Exception:
        return str(left) == str(right)


def _raster_context(
    env: xr.DataArray | xr.Dataset,
    *,
    bands: Sequence[str] | None,
    dtype: str | np.dtype,
    chunks: dict[str, int] | None,
    target_chunk_mb: int,
    align_storage: bool,
    require_inside: bool,
):
    raster = _as_raster_dataarray(env)
    if bands is not None:
        raster = raster.sel(band=list(bands))
    raster = raster.transpose("band", "y", "x")
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
    raster = raster.chunk(chunks)

    x_coordinates = np.asarray(raster["x"].values, dtype=np.float64)
    y_coordinates = np.asarray(raster["y"].values, dtype=np.float64)
    y_lengths = _dimension_chunk_lengths(
        raster, "y", nominal=int(chunks.get("y", raster.sizes["y"]))
    )
    x_lengths = _dimension_chunk_lengths(
        raster, "x", nominal=int(chunks.get("x", raster.sizes["x"]))
    )
    meta = _RouteMeta(
        xmin=float(np.min(x_coordinates)),
        xmax=float(np.max(x_coordinates)),
        ymin=float(np.min(y_coordinates)),
        ymax=float(np.max(y_coordinates)),
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        y_lengths=y_lengths,
        x_lengths=x_lengths,
        y_starts=_chunk_starts(y_lengths),
        x_starts=_chunk_starts(x_lengths),
        sampled_bands=tuple(str(value) for value in raster["band"].values),
        dtype=str(np.dtype(dtype)),
        require_inside=bool(require_inside),
    )
    return meta, raster


def _candidate_chunk_coords(bounds: Bounds, meta: _RouteMeta) -> tuple[tuple[int, int], ...]:
    xmin, ymin, xmax, ymax = bounds
    rows = _nearest_indices(
        meta.y_coordinates, np.asarray([ymin, ymax], dtype=np.float64)
    )
    cols = _nearest_indices(
        meta.x_coordinates, np.asarray([xmin, xmax], dtype=np.float64)
    )
    cy = _chunk_number(rows, meta.y_lengths)
    cx = _chunk_number(cols, meta.x_lengths)
    cy0, cy1 = int(np.min(cy)), int(np.max(cy))
    cx0, cx1 = int(np.min(cx)), int(np.max(cx))
    return tuple(
        (chunk_y, chunk_x)
        for chunk_y in range(cy0, cy1 + 1)
        for chunk_x in range(cx0, cx1 + 1)
    )


def _read_partition(path: str, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns))


def _route_frame_block_local(
    frame: pd.DataFrame,
    *,
    partition_index: int,
    x_col: str,
    y_col: str,
    preserve_cols: tuple[str, ...],
    block_coords: tuple[tuple[int, int], ...],
    meta: _RouteMeta,
    expected_rows: int | None,
) -> _BlockLocalRoute:
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(
            f"Point partition {partition_index} contains {len(frame):,} rows; "
            f"expected {expected_rows:,}."
        )
    point_x = frame[x_col].to_numpy(dtype=np.float64)
    point_y = frame[y_col].to_numpy(dtype=np.float64)
    if meta.require_inside and len(frame):
        outside = (
            (point_x < meta.xmin)
            | (point_x > meta.xmax)
            | (point_y < meta.ymin)
            | (point_y > meta.ymax)
        )
        if np.any(outside):
            raise ValueError(
                f"Point partition {partition_index} contains "
                f"{int(outside.sum()):,} points outside the raster extent."
            )

    row_index = _nearest_indices(meta.y_coordinates, point_y)
    col_index = _nearest_indices(meta.x_coordinates, point_x)
    grouped = _group_positions_by_chunk(
        _chunk_number(row_index, meta.y_lengths),
        _chunk_number(col_index, meta.x_lengths),
        n_x_chunks=len(meta.x_lengths),
    )
    position_dtype = np.uint32 if len(frame) <= np.iinfo(np.uint32).max else np.uint64
    payload_map: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for cy, cx, positions in grouped:
        y_start = int(meta.y_starts[cy])
        x_start = int(meta.x_starts[cx])
        block_width = int(meta.x_lengths[cx])
        local_rows = row_index[positions] - y_start
        local_cols = col_index[positions] - x_start
        max_flat_index = int(meta.y_lengths[cy]) * block_width - 1
        flat_dtype = np.uint32 if max_flat_index <= np.iinfo(np.uint32).max else np.uint64
        payload_map[(int(cy), int(cx))] = (
            positions.astype(position_dtype, copy=False),
            (local_rows * block_width + local_cols).astype(flat_dtype, copy=False),
        )
    unexpected = set(payload_map).difference(block_coords)
    if unexpected:
        raise RuntimeError(
            f"Point partition {partition_index} routed outside its declared bounds; "
            f"unexpected raster chunks: {sorted(unexpected)!r}."
        )
    return _BlockLocalRoute(
        partition_index=int(partition_index),
        rows=len(frame),
        point_x=point_x,
        point_y=point_y,
        preserved=tuple(frame[column].to_numpy(copy=True) for column in preserve_cols),
        payloads=tuple(payload_map.get(coord) for coord in block_coords),
    )


def _select_partition_metadata(route: _BlockLocalRoute) -> _PartitionMetadata:
    return _PartitionMetadata(
        partition_index=route.partition_index,
        rows=route.rows,
        point_x=route.point_x,
        point_y=route.point_y,
        preserved=route.preserved,
    )


def _select_block_payload(route: _BlockLocalRoute, payload_index: int) -> BlockPayload:
    return route.payloads[payload_index]


def _extract_block_local(payload: BlockPayload, block, dtype: str):
    if payload is None:
        return None
    positions, flat_indices = payload
    extracted = _extract_numpy_block_flat(np.asarray(block), flat_indices).astype(
        np.dtype(dtype), copy=False
    )
    return positions, extracted


def _assemble_partition(
    metadata: _PartitionMetadata,
    sampled_bands: tuple[str, ...],
    preserve_cols: tuple[str, ...],
    dtype: str,
    *fragments,
):
    values = np.empty((len(sampled_bands), metadata.rows), dtype=np.dtype(dtype))
    filled = 0
    for fragment in fragments:
        if fragment is None:
            continue
        positions, extracted = fragment
        positions = np.asarray(positions)
        values[:, positions] = np.asarray(extracted, dtype=np.dtype(dtype))
        filled += len(positions)
    if filled != metadata.rows:
        raise RuntimeError(
            f"Point partition {metadata.partition_index} assembled {filled:,} sampled rows; "
            f"expected {metadata.rows:,}."
        )
    out = pd.DataFrame(values.T, columns=list(sampled_bands))
    out["x"] = metadata.point_x
    out["y"] = metadata.point_y
    for column, values_in in zip(preserve_cols, metadata.preserved):
        out[column] = values_in
    return metadata.partition_index, out


def _partition_task(
    partition: PointPartition,
    *,
    partition_index: int,
    x_col: str,
    y_col: str,
    preserve_cols: tuple[str, ...],
    meta: _RouteMeta,
    raster_blocks,
):
    coords = _candidate_chunk_coords(partition.bounds, meta)
    read_columns = tuple(_deduplicate_columns([x_col, y_col, *preserve_cols]))
    frame = dask.delayed(_read_partition, pure=False)(str(partition.path), read_columns)
    route = dask.delayed(_route_frame_block_local, pure=False)(
        frame,
        partition_index=partition_index,
        x_col=x_col,
        y_col=y_col,
        preserve_cols=preserve_cols,
        block_coords=coords,
        meta=meta,
        expected_rows=partition.rows,
    )
    metadata = dask.delayed(_select_partition_metadata, pure=False)(route)
    fragments = []
    for payload_index, (cy, cx) in enumerate(coords):
        payload = dask.delayed(_select_block_payload, pure=False)(route, payload_index)
        fragments.append(
            dask.delayed(_extract_block_local, pure=False)(
                payload, raster_blocks[0, cy, cx], meta.dtype
            )
        )
    return dask.delayed(_assemble_partition, pure=False)(
        metadata,
        meta.sampled_bands,
        preserve_cols,
        meta.dtype,
        *fragments,
    )


def _execution_threads(client) -> int:
    try:
        workers = client.scheduler_info().get("workers", {})
        return max(1, sum(int(worker.get("nthreads", 1)) for worker in workers.values()))
    except Exception:
        return 1


def _default_graph_partitions(client, total_partitions: int) -> int:
    # Empirical point-sampling tuning brackets the useful graph depth below 512:
    # ~100 is appropriate on a 12-thread workstation, while 256 is the measured
    # CoolMUC-4 sweet spot at 112 execution threads. Keep the adaptive lower
    # range while capping large machines before orchestration overhead dominates.
    target = max(32, min(256, _execution_threads(client) * 8))
    return max(1, min(int(total_partitions), int(target)))


def _validate_partition_crs(partitions: Sequence[PointPartition], raster) -> None:
    try:
        raster_crs = raster.rio.crs
    except Exception:
        raster_crs = None
    if raster_crs is None:
        return
    for index, partition in enumerate(partitions):
        if partition.crs is not None and not _same_crs(partition.crs, raster_crs):
            raise ValueError(
                f"Point partition {index} CRS {partition.crs!r} does not match "
                f"raster CRS {raster_crs!r}; transform coordinates before partitioned sampling."
            )


def iter_sample_raster_stack_partitioned(
    partitions: Sequence[PointPartition],
    env: xr.DataArray | xr.Dataset,
    *,
    x_col: str = "x",
    y_col: str = "y",
    bands: Sequence[str] | None = None,
    dtype: str | np.dtype = "float32",
    preserve_cols: Sequence[str] | str | None = None,
    id_cols: Sequence[str] | str | None = None,
    chunks: dict[str, int] | None = None,
    target_chunk_mb: int = 256,
    align_storage: bool = True,
    graph_partitions: int | None = None,
    client=None,
    require_inside: bool = False,
) -> Iterator[SampledPointPartition]:
    """Sample spatial Parquet partitions with a Dask-native block-local graph.

    Completed partitions are yielded in completion order. Use
    ``SampledPointPartition.partition_index`` to recover source order. The point
    files and their bounds must use the raster CRS and be visible to all workers.
    """
    partitions = tuple(partitions)
    if client is None:
        raise ValueError("partitioned point sampling requires an active Dask client.")
    if graph_partitions is not None and int(graph_partitions) <= 0:
        raise ValueError("graph_partitions must be positive when supplied.")
    if not partitions:
        return

    preserve = _deduplicate_columns(
        [*_normalise_columns(preserve_cols), *_normalise_id_cols(id_cols)]
    )
    preserve = [column for column in preserve if column not in {x_col, y_col, "x", "y"}]
    preserve_tuple = tuple(preserve)
    meta, raster = _raster_context(
        env,
        bands=bands,
        dtype=dtype,
        chunks=chunks,
        target_chunk_mb=target_chunk_mb,
        align_storage=align_storage,
        require_inside=require_inside,
    )
    _validate_partition_crs(partitions, raster)
    raster_blocks = np.asarray(raster.data.to_delayed(), dtype=object)
    if raster_blocks.ndim != 3 or raster_blocks.shape[0] != 1:
        raise RuntimeError(
            "Partitioned point sampling expects one band chunk and a 2-D spatial "
            f"raster block grid; observed delayed block shape {raster_blocks.shape}."
        )

    window = (
        _default_graph_partitions(client, len(partitions))
        if graph_partitions is None
        else min(len(partitions), int(graph_partitions))
    )
    try:
        from distributed import as_completed
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Partitioned point sampling requires the hsa[dask] or hsa[hpc] dependencies."
        ) from exc

    for start in range(0, len(partitions), window):
        stop = min(start + window, len(partitions))
        futures = list(
            client.compute(
                [
                    _partition_task(
                        partitions[index],
                        partition_index=index,
                        x_col=x_col,
                        y_col=y_col,
                        preserve_cols=preserve_tuple,
                        meta=meta,
                        raster_blocks=raster_blocks,
                    )
                    for index in range(start, stop)
                ]
            )
        )
        try:
            for future, result in as_completed(
                futures, with_results=True, raise_errors=True
            ):
                try:
                    partition_index, frame = result
                    yield SampledPointPartition(
                        partition_index=int(partition_index),
                        source=partitions[int(partition_index)],
                        frame=frame,
                    )
                finally:
                    try:
                        future.release()
                    except Exception:
                        pass
        finally:
            for future in futures:
                try:
                    future.release()
                except Exception:
                    pass


def sample_raster_stack_partitioned(
    partitions: Sequence[PointPartition],
    env: xr.DataArray | xr.Dataset,
    **kwargs: Any,
) -> pd.DataFrame:
    """Collect partition-native samples into one deterministic dataframe."""
    sampled = list(iter_sample_raster_stack_partitioned(partitions, env, **kwargs))
    if not sampled:
        return pd.DataFrame()
    sampled.sort(key=lambda item: item.partition_index)
    return pd.concat([item.frame for item in sampled], ignore_index=True)
