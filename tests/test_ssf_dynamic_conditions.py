import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hsa.ssf.dynamic_diagnostics import (
    extract_dynamic_condition_snapshots,
    plot_dynamic_condition_comparison,
    summarize_dynamic_conditions,
)


def _field():
    time = pd.date_range("2025-01-15 00:00", periods=24, freq="h")
    latitude = np.array([42.0, 41.0])
    longitude = np.array([0.0, 1.0, 2.0])

    hour = np.arange(24, dtype=float)[:, None, None]
    spatial = np.arange(6, dtype=float).reshape(1, 2, 3)
    upward = hour + spatial
    sshf = -(hour * 3600.0 + spatial * 360.0)
    return xr.Dataset(
        {
            "upward_potential": (
                ("valid_time", "latitude", "longitude"),
                upward,
                {"units": "m s-1"},
            ),
            "sshf": (
                ("valid_time", "latitude", "longitude"),
                sshf,
                {"units": "J m-2"},
            ),
        },
        coords={
            "valid_time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


def test_extract_local_clock_times_and_transform():
    snapshots = extract_dynamic_condition_snapshots(
        _field(),
        variables={
            "upward_potential": "upward_potential",
            "sshf": "thermal_flux_upward",
        },
        times=["2025-01-15 09:00", "2025-01-15 12:00"],
        timezone="Europe/Madrid",
        transforms={"sshf": lambda x: -x / 3600.0},
        bounds=(0.0, 41.0, 2.0, 42.0),
        time_method="nearest",
        time_tolerance="1min",
    )

    assert snapshots.sizes["comparison_time"] == 2
    assert set(snapshots.data_vars) == {
        "upward_potential",
        "thermal_flux_upward",
    }
    # January in Europe/Madrid is UTC+1: 09:00 local -> 08:00 UTC.
    assert pd.Timestamp(snapshots["matched_time_utc"].values[0]) == pd.Timestamp(
        "2025-01-15 08:00"
    )
    assert pd.Timestamp(snapshots["matched_time_utc"].values[1]) == pd.Timestamp(
        "2025-01-15 11:00"
    )
    np.testing.assert_allclose(
        snapshots["thermal_flux_upward"].isel(comparison_time=0).values,
        8.0 + np.arange(6).reshape(2, 3) / 10.0,
    )


def test_summary_reports_spatial_distribution():
    snapshots = extract_dynamic_condition_snapshots(
        _field(),
        variables="upward_potential",
        times=["2025-01-15 08:00+00:00", "2025-01-15 11:00+00:00"],
        timezone=None,
        bounds=(0.0, 41.0, 2.0, 42.0),
        time_tolerance="1min",
    )
    summary = summarize_dynamic_conditions(snapshots, area_weighted=False)

    assert len(summary) == 2
    assert summary["n_cells"].eq(6).all()
    assert summary.iloc[0]["mean"] == pytest.approx(10.5)
    assert summary.iloc[1]["mean"] == pytest.approx(13.5)
    assert summary["fraction_positive"].eq(1.0).all()


def test_derived_variable_receives_selected_dataset():
    snapshots = extract_dynamic_condition_snapshots(
        _field(),
        variables=["upward_potential", "sshf"],
        times="2025-01-15 08:00+00:00",
        timezone=None,
        bounds=(0.0, 41.0, 2.0, 42.0),
        derived={
            "combined_lift_index": lambda ds: ds["upward_potential"]
            + (-ds["sshf"] / 3600.0)
        },
        time_tolerance="1min",
    )
    assert "combined_lift_index" in snapshots
    expected = snapshots["upward_potential"].isel(comparison_time=0).values
    expected = expected + 8.0 + np.arange(6).reshape(2, 3) / 10.0
    np.testing.assert_allclose(
        snapshots["combined_lift_index"].isel(comparison_time=0).values,
        expected,
    )


def test_naive_time_requires_timezone_when_none():
    with pytest.raises(ValueError, match="explicit timezone"):
        extract_dynamic_condition_snapshots(
            _field(),
            variables="upward_potential",
            times="2025-01-15 09:00",
            timezone=None,
        )


def test_plot_returns_figure_summary_and_snapshots():
    pytest.importorskip("matplotlib")
    snapshots = extract_dynamic_condition_snapshots(
        _field(),
        variables="upward_potential",
        times=["2025-01-15 08:00+00:00", "2025-01-15 11:00+00:00"],
        timezone=None,
        bounds=(0.0, 41.0, 2.0, 42.0),
        time_tolerance="1min",
    )
    result = plot_dynamic_condition_comparison(
        snapshots,
        kind="both",
        distribution_kind="ecdf",
        area_weighted=False,
    )
    assert set(result) == {"figure", "axes", "summary", "snapshots"}

    axes = result["axes"]
    assert set(axes) == {"maps", "distributions", "colorbars"}
    assert set(axes["maps"]) == {"upward_potential"}
    assert len(axes["maps"]["upward_potential"]) == 2
    assert axes["distributions"]["upward_potential"] is not None
    assert axes["colorbars"]["upward_potential"] is not None
    assert len(result["summary"]) == 2

    import matplotlib.pyplot as plt

    plt.close(result["figure"])
