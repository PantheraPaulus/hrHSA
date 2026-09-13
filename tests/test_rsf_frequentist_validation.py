import numpy as np
import pandas as pd
import pytest

import hsa.rsf.frequentist_validation as fv


def _fake_diagnostics():
    return {
        "bird-A": {
            "fold": 0,
            "test_pred": pd.DataFrame(),
            "rsf": object(),
            "domain": object(),
        }
    }


def _fake_scores():
    return {
        "used_ranks": np.array(
            [0.05, 0.18, 0.31, 0.44, 0.58, 0.72, 0.86, 0.97],
            dtype=np.float32,
        ),
        "used_data": pd.DataFrame(
            {
                "Timestamp": pd.date_range(
                    "2024-01-04",
                    periods=8,
                    freq="7D",
                    tz="UTC",
                )
            }
        ),
        "available_scores": np.linspace(0.0, 1.0, 200),
    }


@pytest.fixture
def patched_scores(monkeypatch):
    monkeypatch.setattr(
        fv,
        "_prepare_frequentist_scores",
        lambda *args, **kwargs: _fake_scores(),
    )


def test_temporal_random_mode_is_not_supported(patched_scores):
    with pytest.raises(ValueError, match="intentionally unsupported"):
        fv.evaluate_frequentist_loio_uncertainty(
            _fake_diagnostics(),
            methods=("temporal_random",),
            temporal_folds=2,
            n_bins=4,
        )


def test_contiguous_validation_records_exact_weeks(patched_scores):
    result = fv.evaluate_frequentist_loio_uncertainty(
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

    assignments = result["raw"]["bird-A"]["temporal_contiguous"][
        "assignments"
    ]
    per_week = assignments.groupby("temporal_label")["temporal_fold"].nunique()
    assert (per_week == 1).all()


def test_block_bootstrap_keeps_whole_blocks(patched_scores):
    result = fv.evaluate_frequentist_loio_uncertainty(
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
    assert result["config"]["uncertainty_scope"] == "validation_sample_only"


def test_bootstrap_interval_respects_ci_prob(patched_scores):
    result = fv.evaluate_frequentist_loio_uncertainty(
        _fake_diagnostics(),
        methods=("bootstrap",),
        bootstrap_replicates=50,
        bootstrap_block="14D",
        n_bins=4,
        ci_prob=0.80,
        seed=8,
    )

    summary = result["method_summary"].iloc[0]
    assert summary["boyce_lower"] <= summary["boyce_median"]
    assert summary["boyce_median"] <= summary["boyce_upper"]


def test_prepare_scores_rejects_invalid_test_pred_timestamps():
    diagnostic = {
        "test_pred": pd.DataFrame(
            {
                "Timestamp": ["2024-01-01", "not-a-timestamp"],
                "rsf_pred": [1.0, 2.0],
            }
        ),
        "rsf": object(),
        "domain": object(),
    }

    with pytest.raises(
        ValueError,
        match=r"'Timestamp'.*could not be parsed",
    ):
        fv._prepare_frequentist_scores(
            diagnostic,
            n_background=10,
            seed=42,
        )


def test_plot_rejects_empty_curve_summary():
    validation = {
        "curves": pd.DataFrame(),
        "curve_summary": pd.DataFrame(),
        "replicate_summary": pd.DataFrame(),
        "config": {"ci_prob": 0.95},
    }

    with pytest.raises(ValueError, match="curve_summary is empty"):
        fv.plot_frequentist_loio_uncertainty(
            validation,
            "bird-A",
        )
