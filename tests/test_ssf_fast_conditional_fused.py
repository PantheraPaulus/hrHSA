from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hsa.ssf.data import SSFChoiceArrays
from hsa.ssf.fast_conditional import FastConditionalLogitModel


def _arrays(*, offset: bool) -> SSFChoiceArrays:
    rng = np.random.default_rng(20260903)
    n_strata = 64
    n_choices = 7
    n_predictors = 5
    X = rng.normal(size=(n_strata, n_choices, n_predictors))
    chosen = rng.integers(0, n_choices, size=n_strata, dtype=np.int32)
    individual_idx = np.repeat(np.arange(4, dtype=np.int32), n_strata // 4)
    strata = pd.DataFrame(
        {
            "Individual_ID": [f"id-{i}" for i in individual_idx],
            "stratum_id": np.arange(n_strata, dtype=np.int64),
            "individual_idx": individual_idx,
        }
    )
    fixed_offset = (
        rng.normal(scale=0.25, size=(n_strata, n_choices))
        if offset
        else None
    )
    return SSFChoiceArrays(
        X=X,
        chosen=chosen,
        individual_idx=individual_idx,
        individuals=tuple(f"id-{i}" for i in range(4)),
        strata=strata,
        predictors=tuple(f"x{i}" for i in range(n_predictors)),
        n_choices=n_choices,
        offset=fixed_offset,
        offset_name="proposal_offset" if offset else None,
    )


def test_offset_free_model_does_not_allocate_dense_zero_offset():
    model = FastConditionalLogitModel(_arrays(offset=False))
    assert model.offset is None


def test_fused_loglike_and_score_match_separate_evaluations_with_and_without_offset():
    params = np.array([0.35, -0.2, 0.1, 0.05, -0.15], dtype=np.float64)

    for use_offset in (False, True):
        model = FastConditionalLogitModel(_arrays(offset=use_offset))
        expected_loglike = model.loglike(params)
        expected_score = model.score(params)

        actual_loglike, actual_score = model.loglike_and_score(params)

        np.testing.assert_allclose(actual_loglike, expected_loglike, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(actual_score, expected_score, rtol=1e-12, atol=1e-12)


def test_fused_kernel_uses_one_linear_predictor_pass():
    model = FastConditionalLogitModel(_arrays(offset=True))
    params = np.linspace(-0.2, 0.2, model.n_predictors)
    original = model.linear_predictor
    calls = 0

    def counted(values):
        nonlocal calls
        calls += 1
        return original(values)

    model.linear_predictor = counted
    model.loglike_and_score(params)

    assert calls == 1


def test_fused_optimizer_matches_previous_two_pass_objective():
    arrays = _arrays(offset=True)
    old_model = FastConditionalLogitModel(arrays)
    fused_model = FastConditionalLogitModel(arrays)
    start = np.zeros(fused_model.n_predictors, dtype=np.float64)

    def old_objective(beta):
        return -old_model.loglike(beta), -old_model.score(beta)

    def fused_objective(beta):
        loglike, score = fused_model.loglike_and_score(beta)
        return -loglike, -score

    old = minimize(old_objective, start, method="L-BFGS-B", jac=True)
    fused = minimize(fused_objective, start, method="L-BFGS-B", jac=True)

    assert old.success == fused.success
    np.testing.assert_allclose(fused.x, old.x, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(fused.fun, old.fun, rtol=1e-11, atol=1e-11)
