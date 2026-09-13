from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hsa import FeatureSpec
from hsa.compute.raster import _group_positions_by_chunk
from hsa.rsf.bayesian import build_bayesian_rsf_model, prepare_bayesian_rsf_data
from hsa.rsf.inference_validation import (
    compare_posterior_summaries,
    posterior_parameter_summary,
)
from hsa.rsf.model import fit_prepared_rsf, fit_rsf, prepare_rsf_design


def test_vectorized_chunk_grouping_preserves_positions():
    chunk_y = np.array([1, 0, 1, 0, 1, 2, 0], dtype=np.int64)
    chunk_x = np.array([0, 2, 0, 2, 1, 0, 1], dtype=np.int64)

    grouped = _group_positions_by_chunk(chunk_y, chunk_x, n_x_chunks=3)
    observed = {(cy, cx): positions.tolist() for cy, cx, positions in grouped}

    assert observed == {
        (0, 1): [6],
        (0, 2): [1, 3],
        (1, 0): [0, 2],
        (1, 1): [4],
        (2, 0): [5],
    }


def _frequentist_fixture():
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = -1.0 + 0.8 * x1 - 0.5 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    df = pd.DataFrame(
        {
            "used": rng.binomial(1, p),
            "x1": x1,
            "x2": x2,
        }
    )
    return df, FeatureSpec(linear=["x1", "x2"], add_const=True)


def test_frequentist_optimizer_selection_preserves_coefficients():
    df, spec = _frequentist_fixture()
    newton, _, _, _ = fit_rsf(df, spec, method="newton")
    lbfgs, _, _, _ = fit_rsf(df, spec, method="lbfgs")

    np.testing.assert_allclose(
        newton.params.to_numpy(),
        lbfgs.params.to_numpy(),
        rtol=2e-4,
        atol=2e-4,
    )


def test_prepared_design_reuses_exact_matrix_across_optimizers():
    df, spec = _frequentist_fixture()
    prepared = prepare_rsf_design(df, spec)
    newton = fit_prepared_rsf(prepared, method="newton")
    lbfgs = fit_prepared_rsf(prepared, method="lbfgs")

    assert prepared.n_rows == len(df)
    assert prepared.n_columns == 3
    np.testing.assert_allclose(
        newton.params.to_numpy(),
        lbfgs.params.to_numpy(),
        rtol=2e-4,
        atol=2e-4,
    )


def test_posterior_summary_comparison_detects_matching_and_shifted_backends():
    posterior_ref = xr.Dataset(
        {
            "alpha": (("chain", "draw"), np.array([[0.0, 1.0], [0.5, 0.5]])),
            "beta": (
                ("chain", "draw", "coef"),
                np.array([[[1.0, 2.0], [1.2, 1.8]], [[0.9, 2.1], [1.1, 1.9]]]),
            ),
            "eta": (("chain", "draw", "obs"), np.ones((2, 2, 3))),
        }
    )
    posterior_same = posterior_ref.copy(deep=True)
    posterior_shifted = posterior_ref.copy(deep=True)
    posterior_shifted["alpha"] = posterior_shifted["alpha"] + 0.25

    class IData:
        pass

    ref = IData()
    ref.posterior = posterior_ref
    same = IData()
    same.posterior = posterior_same
    shifted = IData()
    shifted.posterior = posterior_shifted

    ref_summary = posterior_parameter_summary(ref)
    same_metrics = compare_posterior_summaries(
        ref_summary, posterior_parameter_summary(same)
    )
    shifted_metrics = compare_posterior_summaries(
        ref_summary, posterior_parameter_summary(shifted)
    )

    assert not any(term == "eta" or term.startswith("eta[") for term in ref_summary["term"])
    assert same_metrics["max_abs_mean_diff"] == pytest.approx(0.0)
    assert shifted_metrics["max_abs_mean_diff"] == pytest.approx(0.25)


def test_bayesian_eta_storage_is_opt_in():
    pytest.importorskip("pymc")
    df = pd.DataFrame(
        {
            "individual-local-identifier": ["A"] * 6 + ["B"] * 6,
            "used": [1, 0, 0, 1, 0, 0] * 2,
            "x1": np.linspace(-2.0, 2.0, 12),
            "x2": np.linspace(1.5, -1.5, 12),
        }
    )
    data = prepare_bayesian_rsf_data(df, ["x1", "x2"])

    default_model = build_bayesian_rsf_model(data, random_intercept=True)
    stored_model = build_bayesian_rsf_model(
        data,
        random_intercept=True,
        store_eta=True,
    )

    assert "eta" not in default_model.named_vars
    assert "eta" in stored_model.named_vars
