import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from hsa.ssf import FrequentistISSF


def _reloc():
    rows = []
    for i, animal in enumerate(("A", "B")):
        for hour in range(3):
            rows.append(
                {
                    "id": animal,
                    "time": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(hours=hour),
                    "geometry": Point(10 + i + 0.01 * hour, 45 + 0.01 * hour),
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def _choices(n_strata_per_id=4, n_choices=3):
    rows = []
    stratum = 0
    for i, animal in enumerate(("A", "B")):
        for local in range(n_strata_per_id):
            start = Point(10 + i + 0.02 * local, 45 + 0.01 * local)
            heat = -1.0 + 0.4 * local + 0.2 * i
            for candidate in range(n_choices):
                rows.append(
                    {
                        "id": animal,
                        "stratum_id": stratum,
                        "candidate_id": candidate,
                        "used": int(candidate == 0),
                        "start_time": pd.Timestamp("2025-01-01", tz="UTC")
                        + pd.Timedelta(hours=local),
                        "end_time": pd.Timestamp("2025-01-01", tz="UTC")
                        + pd.Timedelta(hours=local + 1),
                        "start_geometry": start,
                        "geometry": Point(
                            start.x + 0.01 * (candidate + 1),
                            start.y + 0.005 * candidate,
                        ),
                        "step_length": 800.0 + 700.0 * candidate + 50.0 * local,
                        "turn_angle": -0.8 + 0.7 * candidate,
                        "elevation": 500.0 + 80.0 * candidate + 20.0 * local,
                        "slope": 5.0 + 2.0 * candidate + 0.5 * local,
                        "heat_start": heat,
                        "wind_start_support": -3.0
                        + 3.0 * candidate
                        + 0.1 * local,
                        "proposal_logpdf": -5.0 - 0.4 * candidate,
                    }
                )
            stratum += 1
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def _analysis():
    return FrequentistISSF(
        _reloc(),
        id_col="id",
        timestamp_col="time",
        n_available=2,
        choices=_choices(),
    )


def test_constructor_does_not_require_environment_or_predictors():
    issf = _analysis()
    assert issf.env is None
    assert issf.endpoint_predictors == ()
    assert issf.start_predictors == ()
    assert issf.directional_predictors == ()
    assert issf.model_spec_ is None


def test_set_model_infers_start_and_directional_modifiers():
    issf = _analysis().set_model(
        selection=["elevation", "slope"],
        modifiers={
            "heat_start": "step_length",
            "wind_start_support": "step_length",
        },
    )

    assert issf.start_predictors == ("heat_start",)
    assert issf.directional_predictors == ("wind_start_support",)

    design = issf.prepare_design()
    assert design.predictors == (
        "elevation_z",
        "slope_z",
        "wind_start_support_z",
        "step_length_km",
        "log_step_length",
        "cos_turn_angle",
        "heat_start_x_step_length",
        "heat_start_x_log_step_length",
        "wind_start_support_x_step_length",
        "wind_start_support_x_log_step_length",
    )


def test_set_model_allows_per_modifier_movement_terms():
    issf = _analysis().set_model(
        selection=["elevation"],
        modifiers={
            "heat_start": "step_length",
            "wind_start_support": ["step_length", "turning"],
        },
    )
    design = issf.prepare_design()

    assert "wind_start_support_x_cos_turn_angle" in design.predictors
    assert "heat_start_x_cos_turn_angle" not in design.predictors


def test_endpoint_only_model_is_available_without_clearing_many_components():
    issf = _analysis().set_model(
        selection=["elevation", "slope"],
        movement=False,
    )
    design = issf.prepare_design()

    assert design.predictors == ("elevation_z", "slope_z")
    assert design.offset_col == "proposal_offset"


def test_proposal_correction_can_be_disabled_explicitly():
    issf = _analysis().set_model(
        selection=["elevation", "slope"],
        movement=False,
        proposal_correction=False,
    )
    design = issf.prepare_design()
    assert design.offset_col is None


def test_legacy_set_predictors_accepts_directional_only_modification():
    issf = _analysis()
    issf.set_predictors(
        endpoint_predictors=["elevation"],
        start_predictors=None,
        directional_predictors=["wind_start_support"],
        movement_terms=["step_length_km", "log_step_length", "cos_turn_angle"],
        interaction_terms=["step_length_km"],
    )
    design = issf.prepare_design()
    assert "wind_start_support_x_step_length" in design.predictors


def test_subset_preserves_model_specification_and_complete_strata():
    issf = _analysis().set_model(
        selection=["elevation", "slope"],
        modifiers={
            "heat_start": "step_length",
            "wind_start_support": "step_length",
        },
    )
    small = issf.subset(n_strata_per_id=2, seed=7)

    assert small.model_spec_ == issf.model_spec_
    assert small.directional_predictors == ("wind_start_support",)
    check = small.choices.groupby(["id", "stratum_id"])["used"].agg(
        n_choices="size",
        n_used="sum",
    )
    assert check["n_choices"].eq(3).all()
    assert check["n_used"].eq(1).all()
    assert check.groupby(level=0).size().eq(2).all()
