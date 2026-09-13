import numpy as np
import pandas as pd
import xarray as xr

from hsa.ssf.dynamic_diagnostics import extract_dynamic_condition_snapshots


def _global_0360_field():
    time = pd.date_range("2025-01-15 00:00", periods=3, freq="h")
    latitude = np.arange(40.0, 45.1, 0.5)
    longitude = np.arange(0.0, 360.0, 0.5)
    values = (
        np.arange(len(time), dtype=float)[:, None, None]
        + np.zeros((len(time), len(latitude), len(longitude)), dtype=float)
    )
    return xr.Dataset(
        {
            "sshf": (
                ("valid_time", "latitude", "longitude"),
                values,
            )
        },
        coords={
            "valid_time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


def test_greenwich_crossing_domain_on_0360_field_is_not_treated_as_dateline():
    snapshots = extract_dynamic_condition_snapshots(
        _global_0360_field(),
        variables="sshf",
        times="2025-01-15 01:00+00:00",
        timezone=None,
        bounds=(-3.0, 41.0, 2.0, 44.0),
        time_tolerance="1min",
    )

    lon = snapshots["longitude"].values
    assert lon.min() >= -3.0
    assert lon.max() <= 2.0
    assert np.any(lon < 0)
    assert np.any(lon >= 0)
    assert np.all(np.diff(lon) > 0)


def test_resolution_decimates_before_quick_look_compute():
    snapshots = extract_dynamic_condition_snapshots(
        _global_0360_field(),
        variables="sshf",
        times="2025-01-15 01:00+00:00",
        timezone=None,
        bounds=(-3.0, 40.0, 3.0, 45.0),
        resolution=1.5,
        time_tolerance="1min",
    )

    assert snapshots.attrs["longitude_stride"] == 3
    assert snapshots.attrs["latitude_stride"] == 3
    assert snapshots.sizes["longitude"] <= 5
    assert snapshots.sizes["latitude"] <= 5


def test_max_cells_adds_an_automatic_stride():
    snapshots = extract_dynamic_condition_snapshots(
        _global_0360_field(),
        variables="sshf",
        times="2025-01-15 01:00+00:00",
        timezone=None,
        bounds=(-20.0, 40.0, 20.0, 45.0),
        max_cells=120,
        time_tolerance="1min",
    )

    assert snapshots.attrs["n_spatial_cells"] <= 120
    assert snapshots.attrs["longitude_stride"] > 1
    assert snapshots.attrs["latitude_stride"] > 1
