from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hsa.rsf.bayesian import (
    build_bayesian_rsf_model,
    prepare_bayesian_rsf_data,
)
from hsa.rsf.bayesian_shrinkage import (
    _resolve_horseshoe_config,
    collinearity_diagnostics,
    posterior_beta_correlation,
    predictor_correlation,
    regularized_horseshoe_summary,
)
from hsa.rsf.schemes import BayesianLOIOResult, LeaveOneIndividualOut


def _synthetic_idata():
    draw = np.arange(20, dtype=float)
    beta_a = (draw - draw.mean()) / draw.std()
    beta_b = -beta_a + 0.02 * np.sin(draw)
    beta_c = np.cos(draw)
    beta = np.stack([beta_a, beta_b, beta_c], axis=-1)[None, :, :]

    posterior = xr.Dataset(
        {
            "beta": (("chain", "draw", "predictor"), beta),
        },
        coords={
            "chain": [0],
            "draw": np.arange(len(draw)),
            "predictor": ["a", "b", "c"],
        },
    )
    return SimpleNamespace(posterior=posterior)


def _synthetic_bayes_data():
    return {
        "predictors": ["a", "b", "c"],
        "X": np.array(
            [
                [-2.0, -2.1, 0.0],
                [-1.0, -1.1, 1.0],
                [0.0, 0.1, -1.0],
                [1.0, 1.1, 1.0],
                [2.0, 2.1, 0.0],
            ]
        ),
        "n_trials": np.array([1, 2, 6, 2, 1]),
    }


def test_predictor_correlation_uses_trial_weights():
    result = predictor_correlation(_synthetic_bayes_data())

    assert list(result.index) == ["a", "b", "c"]
    assert np.allclose(np.diag(result), 1.0)
    assert result.loc["a", "b"] > 0.99


def test_collinearity_diagnostics_identifies_posterior_competition():
    result = collinearity_diagnostics(
        _synthetic_idata(),
        _synthetic_bayes_data(),
        min_abs_predictor_corr=0.8,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert {row["predictor_a"], row["predictor_b"]} == {"a", "b"}
    assert row["predictor_corr"] > 0.99
    assert row["posterior_beta_corr"] < -0.99
    assert bool(row["competing_posterior"])


def test_posterior_beta_correlation_returns_named_matrix():
    result = posterior_beta_correlation(_synthetic_idata())

    assert list(result.columns) == ["a", "b", "c"]
    assert result.loc["a", "b"] < -0.99


def test_horseshoe_expected_nonzero_uses_used_locations_as_effective_n():
    config = _resolve_horseshoe_config(
        ["a", "b", "c", "d"],
        used_count=np.array([4, 0, 3, 1]),
        shrinkage={"expected_nonzero": 1},
    )

    expected = (1 / 3) / np.sqrt(8)
    assert np.isclose(config["global_scale"], expected)
    assert config["effective_n"] == 8


def test_horseshoe_effective_n_can_be_overridden():
    config = _resolve_horseshoe_config(
        ["a", "b", "c", "d"],
        used_count=np.array([100, 100]),
        shrinkage={"expected_nonzero": 1, "effective_n": 20},
    )

    expected = (1 / 3) / np.sqrt(20)
    assert np.isclose(config["global_scale"], expected)


def test_regularized_horseshoe_summary_reports_regularization_scales():
    idata = _synthetic_idata()
    posterior = idata.posterior
    shape = posterior["beta"].shape
    local_shape = shape
    scalar_shape = shape[:2]

    posterior["hs_tau"] = (("chain", "draw"), np.full(scalar_shape, 0.2))
    posterior["hs_lambda"] = (
        ("chain", "draw", "predictor"),
        np.full(local_shape, 1.5),
    )
    posterior["hs_lambda_tilde"] = (
        ("chain", "draw", "predictor"),
        np.full(local_shape, 1.2),
    )
    posterior["beta_prior_scale"] = (
        ("chain", "draw", "predictor"),
        np.full(local_shape, 0.24),
    )

    result = regularized_horseshoe_summary(idata)

    assert list(result["predictor"]) == ["a", "b", "c"]
    assert np.allclose(result["tau_median"], 0.2)
    assert np.allclose(result["beta_prior_scale_median"], 0.24)


def test_bayesian_loio_result_exposes_collinearity_diagnostics():
    idata = _synthetic_idata()
    loio = BayesianLOIOResult(
        summary=pd.DataFrame({"heldout_ID": ["bird-A"]}),
        params=pd.DataFrame(),
        boyce_bins=pd.DataFrame(),
        diagnostics={
            "bird-A": {
                "fold": 0,
                "idata": idata,
                "bayes_data": _synthetic_bayes_data(),
            }
        },
        scheme=LeaveOneIndividualOut(heldout=["bird-A"]),
    )

    result = loio.collinearity_diagnostics(
        "bird-A",
        min_abs_predictor_corr=0.8,
    )

    assert len(result) == 1
    assert bool(result.iloc[0]["competing_posterior"])


def test_regularized_horseshoe_builds_named_pymc_variables():
    pytest.importorskip("pymc")

    df = pd.DataFrame(
        {
            "individual-local-identifier": ["A"] * 6 + ["B"] * 6,
            "used": [1, 0, 0, 1, 0, 0] * 2,
            "a": np.linspace(-2, 2, 12),
            "b": np.linspace(-2, 2, 12) + 0.1,
            "c": np.tile([-1.0, 0.0, 1.0], 4),
        }
    )
    data = prepare_bayesian_rsf_data(
        df,
        predictors=["a", "b", "c"],
    )

    model = build_bayesian_rsf_model(
        data,
        predictors=["a", "b", "c"],
        random_intercept=False,
        beta_prior="regularized_horseshoe",
        shrinkage={"expected_nonzero": 1},
    )

    expected = {
        "beta",
        "hs_tau",
        "hs_lambda",
        "hs_c2",
        "hs_beta_raw",
        "hs_lambda_tilde",
        "beta_prior_scale",
    }
    assert expected.issubset(model.named_vars)


def test_regularized_horseshoe_requires_expected_nonzero_or_global_scale():
    pytest.importorskip("pymc")

    df = pd.DataFrame(
        {
            "individual-local-identifier": ["A", "A", "B", "B"],
            "used": [1, 0, 1, 0],
            "a": [-1.0, 0.0, 1.0, 2.0],
            "b": [-0.8, 0.2, 1.2, 2.2],
        }
    )
    data = prepare_bayesian_rsf_data(df, predictors=["a", "b"])

    with pytest.raises(ValueError, match="expected_nonzero.*global_scale"):
        build_bayesian_rsf_model(
            data,
            random_intercept=False,
            beta_prior="regularized_horseshoe",
        )
