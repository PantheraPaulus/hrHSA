import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from hsa.ssf import FrequentistISSF


def _analysis():
    reloc = gpd.GeoDataFrame(
        {
            "id": ["A", "A", "A"],
            "time": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
            "geometry": [
                Point(10.0, 45.0),
                Point(10.1, 45.1),
                Point(10.2, 45.2),
            ],
        },
        geometry="geometry",
        crs=4326,
    )

    rows = []
    for stratum in range(2):
        start = Point(10.0 + 0.1 * stratum, 45.0 + 0.1 * stratum)
        for candidate, delta in enumerate(((0.10, 0.00), (0.05, 0.08), (-0.03, 0.08))):
            rows.append(
                {
                    "id": "A",
                    "stratum_id": stratum,
                    "candidate_id": candidate,
                    "used": int(candidate == 0),
                    "start_time": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(hours=stratum),
                    "end_time": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(hours=stratum + 1),
                    "start_geometry": start,
                    "geometry": Point(start.x + delta[0], start.y + delta[1]),
                    "step_length": 1000.0 + 300.0 * candidate,
                    "turn_angle": -0.5 + 0.5 * candidate,
                    "proposal_logpdf": -5.0 - 0.2 * candidate,
                }
            )
    choices = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    return FrequentistISSF(
        reloc,
        id_col="id",
        timestamp_col="time",
        n_available=2,
        choices=choices,
    )


def _field():
    times = pd.date_range("2025-01-01", periods=5, freq="h").to_numpy()
    latitude = np.array([44.5, 45.0, 45.5])
    longitude = np.array([9.5, 10.0, 10.5, 11.0])
    shape = (len(times), len(latitude), len(longitude))
    temp = np.arange(np.prod(shape), dtype=float).reshape(shape)
    u = np.full(shape, 5.0)
    v = np.full(shape, 1.0)
    return xr.Dataset(
        {
            "temp": (("valid_time", "latitude", "longitude"), temp),
            "u10": (("valid_time", "latitude", "longitude"), u),
            "v10": (("valid_time", "latitude", "longitude"), v),
        },
        coords={
            "valid_time": times,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


def test_unified_dynamic_annotation_handles_endpoint_and_start_times():
    issf = _analysis().annotate_dynamic(
        _field(),
        endpoint={"temp": "temp_end"},
        start={"temp": "temp_start"},
        method="nearest",
        batch_freq=None,
    )

    assert "temp_end" in issf.choices
    assert "temp_start" in issf.choices
    within = issf.choices.groupby(["id", "stratum_id"])["temp_start"].nunique()
    assert within.eq(1).all()


def test_vector_wrapper_creates_candidate_varying_start_support():
    issf = _analysis().annotate_vector(
        _field(),
        u="u10",
        v="v10",
        prefix="wind",
        at="start",
        method="nearest",
        batch_freq=None,
    )

    assert {"wind_u_start", "wind_v_start", "wind_speed_start"}.issubset(
        issf.choices.columns
    )
    assert "wind_start_support" in issf.choices
    variation = issf.choices.groupby(["id", "stratum_id"])[
        "wind_start_support"
    ].nunique()
    assert variation.gt(1).all()

    issf.set_model(
        movement="default",
        modifiers={"wind_start_support": "step_length"},
    )
    assert issf.directional_predictors == ("wind_start_support",)
