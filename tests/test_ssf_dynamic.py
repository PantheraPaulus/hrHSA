import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from hsa.ssf.environment import (
    sample_dynamic_covariates_at_points,
    sample_dynamic_vectors_at_points,
)


def _field():
    time = pd.to_datetime([
        "2020-01-01T00:00:00",
        "2020-01-01T01:00:00",
    ])
    latitude = np.array([1.0, 0.0])
    longitude = np.array([10.0, 11.0])

    sshf = np.array(
        [
            [[-3600.0, -7200.0], [-10800.0, -14400.0]],
            [[-18000.0, -21600.0], [-25200.0, -28800.0]],
        ],
        dtype="float32",
    )
    u10 = np.ones_like(sshf, dtype="float32") * 3.0
    v10 = np.ones_like(sshf, dtype="float32") * 4.0

    return xr.Dataset(
        {
            "sshf": (("valid_time", "latitude", "longitude"), sshf),
            "u10": (("valid_time", "latitude", "longitude"), u10),
            "v10": (("valid_time", "latitude", "longitude"), v10),
        },
        coords={
            "valid_time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


def _points():
    return gpd.GeoDataFrame(
        {
            "end_time": pd.to_datetime(
                [
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T01:00:00Z",
                ]
            )
        },
        geometry=[Point(10.0, 1.0), Point(11.0, 0.0)],
        crs="EPSG:4326",
    )


def test_dynamic_scalar_can_be_renamed_and_transformed():
    out = sample_dynamic_covariates_at_points(
        _points(),
        _field(),
        variables={"sshf": "sensible_heat_upward_wm2"},
        transforms={"sshf": lambda values: -values / 3600.0},
        method="nearest",
    )

    assert "sshf" not in out.columns
    assert np.allclose(
        out["sensible_heat_upward_wm2"].to_numpy(),
        [1.0, 8.0],
    )


def test_dynamic_sampler_accepts_multiple_scalar_variables():
    out = sample_dynamic_covariates_at_points(
        _points(),
        _field(),
        variables=["sshf", "u10"],
        method="nearest",
    )

    assert np.allclose(out["sshf"].to_numpy(), [-3600.0, -28800.0])
    assert np.allclose(out["u10"].to_numpy(), [3.0, 3.0])


def test_vector_sampler_delegates_and_derives_speed():
    out = sample_dynamic_vectors_at_points(
        _points(),
        _field(),
        method="nearest",
    )

    assert np.allclose(out["u10"], 3.0)
    assert np.allclose(out["v10"], 4.0)
    assert np.allclose(out["wind_speed"], 5.0)


def test_unnamed_dataarray_supported_for_single_requested_variable():
    da = _field()["sshf"].rename(None)
    out = sample_dynamic_covariates_at_points(
        _points(),
        da,
        variables="sshf",
        method="nearest",
    )
    assert np.allclose(out["sshf"].to_numpy(), [-3600.0, -28800.0])
