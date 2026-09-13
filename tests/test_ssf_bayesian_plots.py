from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hsa.ssf.bayesian import BayesianSSFFit
from hsa.ssf.bayesian_diagnostics import (
    plot_bayesian_ssf_diagnostics,
    plot_bayesian_ssf_forests,
    plot_bayesian_ssf_trace,
)
from hsa.ssf.data import SSFChoiceArrays


def _idata():
    rng = np.random.default_rng(4)
    predictors = ["elevation_z", "slope_z"]
    individuals = ["A", "B", "C"]
    posterior = xr.Dataset(
        {
            "mu_beta": (
                ("chain", "draw", "predictor"),
                rng.normal(size=(2, 20, 2)),
            ),
            "sigma_beta": (
                ("chain", "draw", "predictor"),
                np.abs(rng.normal(size=(2, 20, 2))),
            ),
            "beta": (
                ("chain", "draw", "individual", "predictor"),
                rng.normal(size=(2, 20, 3, 2)),
            ),
        },
        coords={
            "predictor": predictors,
            "individual": individuals,
        },
    )
    return SimpleNamespace(posterior=posterior)


class _FakeArviZ:
    trace_calls = []
    forest_calls = []

    @classmethod
    def reset(cls):
        cls.trace_calls = []
        cls.forest_calls = []

    @classmethod
    def plot_trace_dist(cls, idata, **kwargs):
        cls.trace_calls.append(kwargs)
        return {"kind": "trace", **kwargs}

    @classmethod
    def plot_forest(cls, idata, **kwargs):
        cls.forest_calls.append(kwargs)
        return {"kind": "forest", **kwargs}


@pytest.fixture(autouse=True)
def _patch_arviz(monkeypatch):
    from hsa.ssf import bayesian_diagnostics

    _FakeArviZ.reset()
    monkeypatch.setattr(bayesian_diagnostics, "_arviz", lambda: _FakeArviZ)


def test_bayesian_ssf_trace_is_compact_by_default():
    out = plot_bayesian_ssf_trace(_idata())
    assert out["var_names"] == ["mu_beta", "sigma_beta"]
    assert "coords" not in out


def test_bayesian_ssf_trace_can_select_individuals():
    out = plot_bayesian_ssf_trace(
        _idata(),
        include_individual=True,
        individuals=["A", "C"],
    )
    assert out["var_names"] == ["mu_beta", "sigma_beta", "beta"]
    assert out["coords"] == {"individual": ["A", "C"]}


def test_bayesian_ssf_forests_keep_hierarchy_separate():
    out = plot_bayesian_ssf_forests(
        _idata(),
        predictors=["elevation_z", "slope_z"],
        ci_prob=0.89,
    )
    assert out["population"]["var_names"] == ["mu_beta"]
    assert out["heterogeneity"]["var_names"] == ["sigma_beta"]
    assert set(out["individual"]) == {"elevation_z", "slope_z"}
    assert out["individual"]["elevation_z"]["coords"] == {
        "predictor": ["elevation_z"]
    }
    assert all(call["ci_probs"] == (0.5, 0.89) for call in _FakeArviZ.forest_calls)


def test_bayesian_ssf_diagnostics_returns_rsf_like_structure():
    out = plot_bayesian_ssf_diagnostics(
        _idata(),
        forest_predictors=["elevation_z"],
    )
    assert set(out) == {
        "trace",
        "population_forest",
        "heterogeneity_forest",
        "individual_forests",
    }
    assert set(out["individual_forests"]) == {"elevation_z"}


def test_bayesian_ssf_forest_rejects_unknown_predictor():
    with pytest.raises(ValueError, match="not present"):
        plot_bayesian_ssf_forests(_idata(), predictors=["not_a_predictor"])


def test_fitted_object_accepts_raw_predictor_names(monkeypatch):
    arrays = SSFChoiceArrays(
        X=np.zeros((1, 2, 2), dtype="float32"),
        chosen=np.array([0], dtype="int32"),
        individual_idx=np.array([0], dtype="int32"),
        individuals=("A",),
        strata=pd.DataFrame(
            {"id": ["A"], "stratum_id": [0], "individual_idx": [0]}
        ),
        predictors=("elevation_z", "slope_z"),
        n_choices=2,
    )
    fit = BayesianSSFFit(
        model=None,
        idata=_idata(),
        arrays=arrays,
        data=pd.DataFrame(),
        raw_predictors=("elevation", "slope"),
        predictors=("elevation_z", "slope_z"),
        scaling={},
        id_col="id",
    )

    captured = {}

    def fake_forests(idata, *, predictors=None, **kwargs):
        captured["predictors"] = predictors
        return {"population": None, "heterogeneity": None, "individual": {}}

    monkeypatch.setattr("hsa.ssf.bayesian.plot_bayesian_ssf_forests", fake_forests)
    fit.plot_forest(predictors=["elevation", "slope_z"])
    assert captured["predictors"] == ["elevation_z", "slope_z"]
