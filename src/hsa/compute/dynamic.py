"""Spatiotemporal tiling for large dynamic environmental sampling workloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hsa._time import require_timezone_aware


def sample_dynamic_covariates_tiled(
    points: gpd.GeoDataFrame,
    field: xr.Dataset | xr.DataArray,
    *,
    variables: Sequence[str] | Mapping[str, str] | str,
    time_col: str = "end_time",
    temporal_tile: str | None = "M",
    spatial_tile_degrees: float = 2.0,
    transforms: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Sample dynamic fields using joint temporal and spatial tiles.

    The core SSF sampler already bounds each remote Xarray read in time and space.
    Its default monthly grouping can nevertheless create a very large geographic
    envelope when animals are widely separated. This wrapper first partitions the
    queries into ``time tile × lon tile × lat tile`` groups and then delegates each
    group to the well-tested core sampler.

    ``spatial_tile_degrees`` controls only query routing; interpolation still uses
    the original coordinates and field resolution. A value between 1 and 5 degrees
    is a useful starting point for ERA5/ERA5-Land and should be benchmarked for the
    target object store.
    """
    if points.crs is None:
        raise ValueError("points.crs is None.")
    if time_col not in points:
        raise KeyError(f"{time_col!r} not found in points.")
    if spatial_tile_degrees <= 0:
        raise ValueError("spatial_tile_degrees must be positive.")
    if "batch_freq" in kwargs:
        raise TypeError(
            "sample_dynamic_covariates_tiled manages batching itself; use temporal_tile=... "
            "instead of batch_freq."
        )

    # Imported lazily so ``import hsa.compute`` does not eagerly initialize the
    # complete hsa.ssf public package. This keeps the compute namespace lightweight
    # and avoids cross-package initialization cycles.
    from hsa.ssf.environment import sample_dynamic_covariates_at_points

    ll = points.to_crs(4326)
    query_time = require_timezone_aware(
        points[time_col],
        name=time_col,
        to_utc=True,
    )
    if query_time.isna().any():
        raise ValueError(f"{time_col!r} contains invalid timestamps.")

    longitude = ll.geometry.x.to_numpy(dtype=float)
    latitude = ll.geometry.y.to_numpy(dtype=float)
    tile_x = np.floor((longitude + 180.0) / spatial_tile_degrees).astype(np.int64)
    tile_y = np.floor((latitude + 90.0) / spatial_tile_degrees).astype(np.int64)

    routing = pd.DataFrame(
        {
            "_position": np.arange(len(points), dtype=np.int64),
            "_tile_x": tile_x,
            "_tile_y": tile_y,
        },
        index=points.index,
    )
    if temporal_tile is None:
        routing["_tile_time"] = 0
    else:
        routing["_tile_time"] = (
            query_time.dt.tz_localize(None).dt.to_period(temporal_tile)
        )

    pieces: list[gpd.GeoDataFrame] = []
    for _, group in routing.groupby(
        ["_tile_time", "_tile_x", "_tile_y"],
        sort=True,
    ):
        subset = points.iloc[group["_position"].to_numpy()].copy()
        sampled = sample_dynamic_covariates_at_points(
            subset,
            field,
            variables=variables,
            time_col=time_col,
            transforms=transforms,
            batch_freq=None,
            **kwargs,
        )
        pieces.append(sampled)

    if not pieces:
        return points.copy()
    out = pd.concat(pieces).sort_index()
    return gpd.GeoDataFrame(out, geometry="geometry", crs=points.crs)
