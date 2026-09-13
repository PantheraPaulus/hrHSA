import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, box

import hsa.rsf.bayesian_validation as bv
from hsa.rsf.bayesian_validation import evaluate_bayesian_loio_uncertainty


def _fake_diagnostics():
    used_scores = np.array(
        [
            [-2.0, -1.5, -1.0, -0.2, 0.1, 0.5, 1.1, 1.8],
            [-1.8, -1.4, -0.8, -0.1, 0.2, 0.7, 1.0, 2.0],
        ]
    )
    available = np.linspace(-2.5, 2.5, 200)
    available_scores = np.vstack([available, available * 0.95])
    used_data = pd.DataFrame(
        {
            "Timestamp": pd.date_range(
                "2024-01-04",
                periods=8,
                freq="7D",
                tz="UTC",
            )
        }
    )
    return {
        "bird-A": {
            "fold": 0,
            "scores": {
                "used_scores": used_scores,
                "available_scores": available_scores,
                "used_data": used_data,
            },
        }
    }


def test_temporal_random_mode_is_removed():
    with pytest.raises(ValueError, match="removed as redundant"):
        evaluate_bayesian_loio_uncertainty(
            _fake_diagnostics(),
            methods=("temporal_random",),
            temporal_folds=2,
            n_bins=4,
        )


def test_contiguous_validation_records_exact_weeks():
    result = evaluate_bayesian_loio_uncertainty(
        _fake_diagnostics(),
        methods=("temporal_contiguous",),
        temporal_folds=2,
        temporal_unit="W",
        n_bins=4,
    )

    rows = result["replicate_summary"]
    assert len(rows) == 2
    assert rows["replicate_label"].str.match(r"F0[01] \| 2024-W").all()
    assert rows["heldout_units"].notna().all()
    assert rows["heldout_units"].str.contains("2024-W").all()
    assert rows["n_temporal_units"].sum() == 8

    assignments = result["raw"]["bird-A"]["temporal_contiguous"]["assignments"]
    per_week = assignments.groupby("temporal_label")["temporal_fold"].nunique()
    assert (per_week == 1).all()


def test_block_bootstrap_keeps_whole_blocks():
    result = evaluate_bayesian_loio_uncertainty(
        _fake_diagnostics(),
        methods=("bootstrap",),
        bootstrap_replicates=25,
        bootstrap_block="14D",
        n_bins=4,
        seed=7,
    )
    rows = result["replicate_summary"]
    assert len(rows) == 25
    assert (rows["n_used"] % 2 == 0).all()
    assert (rows["n_temporal_units"] == 4).all()
    assert set(rows["method"]) == {"bootstrap"}


def test_prepare_bayesian_boyce_samples_only_predictor_bands(monkeypatch):
    predictors = ["elevation", "slope"]

    used = gpd.GeoDataFrame(
        {
            "Timestamp": pd.date_range(
                "2024-01-01",
                periods=2,
                freq="D",
                tz="UTC",
            )
        },
        geometry=[
            Point(0.0, 0.0),
            Point(1.0, 1.0),
        ],
        crs="EPSG:3857",
    )
    domain = gpd.GeoDataFrame(
        geometry=[box(-1.0, -1.0, 2.0, 2.0)],
        crs=used.crs,
    )
    env = xr.DataArray(
        np.zeros((3, 2, 2), dtype=np.float32),
        dims=("band", "y", "x"),
        coords={
            "band": ["elevation", "slope", "unused"],
            "y": [0.0, 1.0],
            "x": [0.0, 1.0],
        },
    )

    available_points = gpd.GeoDataFrame(
        {"used": [False, False]},
        geometry=[
            Point(0.0, 1.0),
            Point(1.0, 0.0),
        ],
        crs=used.crs,
    )
    monkeypatch.setattr(
        bv,
        "sample_available_points",
        lambda *args, **kwargs: available_points.copy(),
    )

    sampled_band_calls = []

    def fake_sample_frame(samples, env_arg, **kwargs):
        sampled_band_calls.append(
            (
                list(env_arg["band"].values),
                list(kwargs["bands"]),
            )
        )
        n = len(samples)
        frame = pd.DataFrame(
            {
                "elevation": np.linspace(1.0, 2.0, n),
                "slope": np.linspace(3.0, 4.0, n),
            }
        )
        if "used" in samples.columns:
            frame["used"] = samples["used"].to_numpy()
        if "Timestamp" in samples.columns:
            frame["Timestamp"] = samples["Timestamp"].to_numpy()
        return frame

    monkeypatch.setattr(
        bv,
        "_sample_frame",
        fake_sample_frame,
    )
    monkeypatch.setattr(
        bv,
        "_extract_population_beta_draws",
        lambda *args, **kwargs: np.array(
            [
                [0.5, -0.2],
                [0.6, -0.1],
            ],
            dtype=np.float32,
        ),
    )

    result = bv.prepare_bayesian_boyce_scores(
        used,
        env,
        idata=object(),
        meta={
            "elevation": {"mean": 0.0, "sd": 1.0},
            "slope": {"mean": 0.0, "sd": 1.0},
        },
        predictors=predictors,
        domain=domain,
        n_background=2,
        n_draws=2,
        seed=42,
    )

    assert sampled_band_calls == [
        (predictors, predictors),
        (predictors, predictors),
    ]
    assert list(result["env"]["band"].values) == predictors
