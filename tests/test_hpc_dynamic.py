from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hsa.compute import sample_dynamic_covariates_tiled
from hsa.ssf.environment import sample_dynamic_covariates_at_points


def test_tiled_dynamic_sampling_matches_core_sampler():
    # Dynamic-field coordinates are UTC-like xarray timestamps, while telemetry
    # timestamps entering hrHSA must be explicitly timezone-aware.
    field_time = pd.date_range("2025-01-01", periods=4, freq="h")
    point_time = field_time.tz_localize("UTC")
    latitude = np.array([0.0, 1.0, 2.0])
    longitude = np.array([0.0, 1.0, 2.0, 3.0])

    t = np.arange(len(field_time), dtype=float)[:, None, None]
    lat = latitude[None, :, None]
    lon = longitude[None, None, :]
    values = (10.0 * t + 2.0 * lat + lon).astype("float32")
    field = xr.Dataset(
        {
            "temperature": (
                ("valid_time", "latitude", "longitude"),
                values,
            )
        },
        coords={
            "valid_time": field_time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    points = gpd.GeoDataFrame(
        {
            "end_time": point_time,
        },
        geometry=gpd.points_from_xy(
            [0.1, 0.9, 2.1, 2.9],
            [0.1, 1.1, 0.9, 1.9],
        ),
        crs="EPSG:4326",
    )

    reference = sample_dynamic_covariates_at_points(
        points,
        field,
        variables="temperature",
        time_col="end_time",
        method="nearest",
        batch_freq=None,
        spatial_margin=0.25,
        time_margin="1h",
    )
    tiled = sample_dynamic_covariates_tiled(
        points,
        field,
        variables="temperature",
        time_col="end_time",
        temporal_tile="D",
        spatial_tile_degrees=1.0,
        method="nearest",
        spatial_margin=0.25,
        time_margin="1h",
    )

    np.testing.assert_allclose(
        reference["temperature"].to_numpy(),
        tiled["temperature"].to_numpy(),
        rtol=0,
        atol=0,
    )
