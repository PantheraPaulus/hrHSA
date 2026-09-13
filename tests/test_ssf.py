import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from hsa.ssf.choice_sets import (
    build_observed_ssf_steps,
    sample_available_steps,
)
from hsa.ssf.data import (
    build_ssf_choice_arrays,
    complete_ssf_strata,
    score_choice_probabilities,
)
from hsa.ssf.diagnostics import (
    canonical_ciif,
    conditional_information,
)
from hsa.ssf.environment import add_vector_support_covariates
from hsa.ssf.validation import make_temporal_block_split


def _movement():
    angle = gpd.GeoDataFrame(
        {
            "Individual_ID": ["A", "A"],
            "Timestamp": pd.to_datetime(
                ["2020-01-01T01:00Z", "2020-01-01T02:00Z"]
            ),
            "next_timestamp": pd.to_datetime(
                ["2020-01-01T02:00Z", "2020-01-01T03:00Z"]
            ),
            "previous_location": [Point(0, 0), Point(1, 0)],
            "next_position": [Point(2, 0), Point(3, 0)],
        },
        geometry=[Point(1, 0), Point(2, 0)],
        crs="EPSG:3857",
    )
    summary = pd.DataFrame(
        {
            "Individual_ID": ["A"],
            "step_distribution": ["exp"],
            "step_params": [(1.0,)],
            "angle_distribution": ["vonmises"],
            "angle_params": [(2.0, 0.0)],
        }
    )
    return {"angle_df": angle, "summary": summary}


def test_observed_steps_have_expected_geometry_semantics():
    observed = build_observed_ssf_steps(
        _movement(),
        id_col="Individual_ID",
    )
    assert len(observed) == 2
    assert np.allclose(observed["step_length"], 1.0)
    assert np.allclose(observed["turn_angle"], 0.0)
    assert observed["used"].eq(1).all()


def test_available_steps_share_start_and_respect_truncation():
    movement = _movement()
    observed = build_observed_ssf_steps(
        movement,
        id_col="Individual_ID",
    )
    choices = sample_available_steps(
        observed,
        movement["summary"],
        id_col="Individual_ID",
        n_available=4,
        max_speed_kmh=pd.Series({"A": 0.01}),
        seed=2,
    )
    counts = choices.groupby("stratum_id")["used"].agg(
        n="size",
        n_used="sum",
    )
    assert counts["n"].eq(5).all()
    assert counts["n_used"].eq(1).all()
    assert (
        choices["step_length"]
        <= choices["max_step_length"] + 1e-10
    ).all()
    assert choices.groupby("stratum_id")["start_x"].nunique().eq(1).all()
    assert choices.groupby("stratum_id")["start_y"].nunique().eq(1).all()


def test_complete_ssf_strata_drops_whole_incomplete_choice_set():
    df = pd.DataFrame(
        {
            "id": ["A"] * 6,
            "stratum_id": [0, 0, 0, 1, 1, 1],
            "used": [1, 0, 0, 1, 0, 0],
            "x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        }
    )
    complete = complete_ssf_strata(
        df,
        predictors=["x"],
        id_col="id",
        expected_n_choices=3,
    )
    assert complete["stratum_id"].unique().tolist() == [1]
    assert len(complete) == 3


def test_choice_array_preserves_candidate_order_and_chosen_index():
    rows = []
    for stratum in range(2):
        for candidate in range(3):
            rows.append(
                {
                    "id": "A",
                    "stratum_id": stratum,
                    "candidate_id": candidate,
                    "used": int(candidate == 1),
                    "x_z": 10 * stratum + candidate,
                }
            )
    df = pd.DataFrame(rows).sample(frac=1, random_state=1)
    arrays = build_ssf_choice_arrays(
        df,
        id_col="id",
        predictors=["x_z"],
    )
    assert arrays.X.shape == (2, 3, 1)
    assert np.array_equal(arrays.chosen, [1, 1])
    assert np.array_equal(
        arrays.X[:, :, 0],
        [[0, 1, 2], [10, 11, 12]],
    )


def test_uniform_choice_has_zero_gain_and_average_tie_rank():
    probability = np.full((5, 4), 0.25)
    per, summary = score_choice_probabilities(
        probability,
        np.array([0, 1, 2, 3, 0]),
    )
    assert np.allclose(per["log_score_gain"], 0.0)
    assert np.allclose(per["choice_rank"], 2.5)
    assert np.isclose(summary["predictive_advantage"], 1.0)
    assert np.isclose(summary["top_1"], 0.0)


def test_conditional_information_matches_weighted_variance():
    df = pd.DataFrame(
        {
            "id": ["A", "A", "A"],
            "stratum_id": [0, 0, 0],
            "x": [-1.0, 0.0, 1.0],
        }
    )
    info = conditional_information(
        df,
        beta=[0.0],
        predictors=["x"],
        id_col="id",
    )
    assert np.isclose(info.loc[0, "information"], 2.0 / 3.0)


class DummyModel:
    def __init__(self, information):
        self.information = np.asarray(information, dtype=float)

    def hessian(self, params):
        return -self.information


def test_ciif_is_one_for_unique_information():
    out = canonical_ciif(
        DummyModel([[4.0, 0.0], [0.0, 9.0]]),
        [0.0, 0.0],
        predictor_names=["a", "b"],
    )
    assert np.allclose(out["ciif"], 1.0)


def test_ciif_increases_with_shared_information():
    out = canonical_ciif(
        DummyModel([[1.0, 0.5], [0.5, 1.0]]),
        [0.0, 0.0],
        predictor_names=["a", "b"],
    )
    assert np.allclose(
        out["ciif"],
        1 / (1 - 0.5**2),
    )


def test_vector_support_uses_true_east_north_components():
    choices = gpd.GeoDataFrame(
        {
            "start_geometry": [Point(0.0, 0.0)],
            "u10": [0.0],
            "v10": [2.0],
            "wind_speed": [2.0],
        },
        geometry=[Point(0.0, 1.0)],
        crs="EPSG:4326",
    )
    out = add_vector_support_covariates(choices)
    assert np.isclose(out.loc[0, "wind_support"], 2.0, atol=1e-8)
    assert np.isclose(out.loc[0, "crosswind"], 0.0, atol=1e-8)
    assert np.isclose(out.loc[0, "wind_alignment"], 1.0, atol=1e-8)


def test_temporal_split_is_contiguous_and_embargoed():
    rows = []
    for hour in range(50):
        for candidate in range(2):
            rows.append(
                {
                    "id": "A",
                    "stratum_id": hour,
                    "candidate_id": candidate,
                    "used": int(candidate == 0),
                    "start_time": (
                        pd.Timestamp("2020-01-01", tz="UTC")
                        + pd.Timedelta(hours=hour)
                    ),
                }
            )
    split = make_temporal_block_split(
        pd.DataFrame(rows),
        id_col="id",
        n_blocks=5,
        holdout_block=2,
        embargo="2h",
    )
    test_times = split["test_keys"]["start_time"]
    assert (
        test_times.max() - test_times.min()
        == pd.Timedelta(hours=9)
    )
    train_times = split["train_keys"]["start_time"]
    assert not (
        (train_times >= test_times.min() - pd.Timedelta("2h"))
        & (train_times <= test_times.max() + pd.Timedelta("2h"))
    ).any()


def test_hierarchical_ssf_model_builds_when_pymc_installed():
    pytest.importorskip("pymc")
    from hsa.ssf.bayesian import build_hierarchical_ssf_model

    rows = []
    for animal in ("A", "B"):
        for stratum in range(3):
            for candidate in range(3):
                rows.append(
                    {
                        "id": animal,
                        "stratum_id": f"{animal}-{stratum}",
                        "candidate_id": candidate,
                        "used": int(candidate == 0),
                        "x_z": float(candidate),
                    }
                )
    arrays = build_ssf_choice_arrays(
        pd.DataFrame(rows),
        id_col="id",
        predictors=["x_z"],
    )
    model = build_hierarchical_ssf_model(arrays)
    logps = model.point_logps()
    assert all(np.isfinite(value) for value in logps.values())
