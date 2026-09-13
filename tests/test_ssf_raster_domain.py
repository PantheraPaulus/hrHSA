import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from hsa.sampling import sample_raster_stack
from hsa.ssf.domain import redraw_available_steps_inside_raster
from hsa.ssf.environment import check_raster_coverage


def test_sample_raster_stack_accepts_single_variable_dataset():
    raster = xr.DataArray(
        np.arange(12, dtype="float32").reshape(1, 3, 4),
        dims=("band", "y", "x"),
        coords={
            "band": ["elevation"],
            "y": [2.0, 1.0, 0.0],
            "x": [0.0, 1.0, 2.0, 3.0],
        },
    )
    dataset = raster.to_dataset(name="terrain")
    points = gpd.GeoDataFrame(
        geometry=[Point(1.0, 1.0), Point(3.0, 0.0)],
        crs="EPSG:3857",
    )

    sampled = sample_raster_stack(
        points,
        dataset,
        bands=["elevation"],
    )

    assert sampled["elevation"].tolist() == [5.0, 11.0]


def _movement():
    return {
        "step_df": pd.DataFrame(
            {"Individual_ID": ["A"], "speed_kmh": [1.0]}
        ),
        "summary": pd.DataFrame(
            {
                "Individual_ID": ["A"],
                "step_distribution": ["exp"],
                "step_params": [(0.1,)],
                "angle_distribution": ["vonmises"],
                "angle_params": [(20.0, 0.0)],
            }
        ),
    }


def _env():
    return xr.DataArray(
        np.zeros((1, 3, 4), dtype="float32"),
        dims=("band", "y", "x"),
        coords={
            "band": ["elevation"],
            "y": [1.0, 0.0, -1.0],
            "x": [0.0, 1.0, 2.0, 3.0],
        },
    )


def _observed_row(*, stratum_id=0, endpoint=Point(2.0, 0.0)):
    return {
        "Individual_ID": "A",
        "stratum_id": stratum_id,
        "candidate_id": 0,
        "used": 1,
        "start_time": pd.Timestamp("2020-01-01T01:00Z")
        + pd.Timedelta(hours=stratum_id),
        "end_time": pd.Timestamp("2020-01-01T02:00Z")
        + pd.Timedelta(hours=stratum_id),
        "dt_h": 1.0,
        "start_x": 1.0,
        "start_y": 0.0,
        "start_geometry": Point(1.0, 0.0),
        "incoming_heading": 0.0,
        "step_length": 1.0,
        "turn_angle": 0.0,
        "heading": 0.0,
        "geometry": endpoint,
    }


def test_outside_available_alternative_is_redrawn_inside_raster():
    observed = _observed_row()
    outside = {
        **observed,
        "candidate_id": 1,
        "used": 0,
        "step_length": 9.0,
        "geometry": Point(10.0, 0.0),
    }
    choices = gpd.GeoDataFrame(
        [observed, outside],
        geometry="geometry",
        crs="EPSG:3857",
    )

    constrained, diagnostics = redraw_available_steps_inside_raster(
        choices,
        _movement(),
        _env(),
        id_col="Individual_ID",
        n_available=1,
        speed_margin=1.05,
        seed=5,
    )

    assert check_raster_coverage(constrained, _env()).all()
    assert (
        constrained.groupby("stratum_id")["used"]
        .agg(["size", "sum"])
        .iloc[0]
        .tolist()
        == [2, 1]
    )
    assert diagnostics["n_outside_initial"] == 1
    assert diagnostics["n_available_outside_initial"] == 1
    assert diagnostics["n_observed_outside_initial"] == 0
    assert diagnostics["n_available_replaced_total"] == 1
    assert diagnostics["n_outside_final"] == 0


def test_candidate_wise_rejection_keeps_valid_alternatives():
    observed = _observed_row()
    valid = {
        **observed,
        "candidate_id": 1,
        "used": 0,
        "step_length": 0.5,
        "geometry": Point(1.5, 0.0),
    }
    outside = {
        **observed,
        "candidate_id": 2,
        "used": 0,
        "step_length": 9.0,
        "geometry": Point(10.0, 0.0),
    }
    choices = gpd.GeoDataFrame(
        [observed, valid, outside],
        geometry="geometry",
        crs="EPSG:3857",
    )

    constrained, diagnostics = redraw_available_steps_inside_raster(
        choices,
        _movement(),
        _env(),
        id_col="Individual_ID",
        n_available=2,
        speed_margin=1.05,
        seed=5,
    )

    kept_valid = constrained.loc[
        constrained["candidate_id"].eq(1), "geometry"
    ].iloc[0]
    assert kept_valid.equals(Point(1.5, 0.0))
    assert check_raster_coverage(constrained, _env()).all()
    assert diagnostics["n_available_replaced_total"] == 1
    assert (
        constrained.groupby("stratum_id")["used"]
        .agg(["size", "sum"])
        .iloc[0]
        .tolist()
        == [3, 1]
    )


def test_observed_outside_raises_by_default():
    observed_inside = _observed_row(stratum_id=0)
    available_inside = {
        **observed_inside,
        "candidate_id": 1,
        "used": 0,
        "geometry": Point(1.5, 0.0),
        "step_length": 0.5,
    }
    observed_outside = _observed_row(
        stratum_id=1,
        endpoint=Point(10.0, 0.0),
    )
    available_for_outside = {
        **observed_outside,
        "candidate_id": 1,
        "used": 0,
        "geometry": Point(2.5, 0.0),
        "step_length": 1.5,
    }
    choices = gpd.GeoDataFrame(
        [
            observed_inside,
            available_inside,
            observed_outside,
            available_for_outside,
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="observed_outside='exclude'"):
        redraw_available_steps_inside_raster(
            choices,
            _movement(),
            _env(),
            id_col="Individual_ID",
            n_available=1,
        )


def test_observed_outside_can_exclude_whole_stratum():
    observed_inside = _observed_row(stratum_id=0)
    available_inside = {
        **observed_inside,
        "candidate_id": 1,
        "used": 0,
        "geometry": Point(1.5, 0.0),
        "step_length": 0.5,
    }
    observed_outside = _observed_row(
        stratum_id=1,
        endpoint=Point(10.0, 0.0),
    )
    available_for_outside = {
        **observed_outside,
        "candidate_id": 1,
        "used": 0,
        "geometry": Point(2.5, 0.0),
        "step_length": 1.5,
    }
    choices = gpd.GeoDataFrame(
        [
            observed_inside,
            available_inside,
            observed_outside,
            available_for_outside,
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )

    constrained, diagnostics = redraw_available_steps_inside_raster(
        choices,
        _movement(),
        _env(),
        id_col="Individual_ID",
        n_available=1,
        observed_outside="exclude",
    )

    assert constrained["stratum_id"].unique().tolist() == [0]
    assert check_raster_coverage(constrained, _env()).all()
    assert (
        constrained.groupby("stratum_id")["used"]
        .agg(["size", "sum"])
        .iloc[0]
        .tolist()
        == [2, 1]
    )
    assert diagnostics["observed_outside_policy"] == "exclude"
    assert diagnostics["n_observed_outside_initial"] == 1
    assert diagnostics["n_observed_strata_excluded"] == 1
    assert diagnostics["observed_strata_excluded_by_id"] == {"A": 1}
    assert diagnostics["excluded_stratum_ids"] == [1]
    assert diagnostics["n_outside_final"] == 0
