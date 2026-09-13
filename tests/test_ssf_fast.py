import numpy as np
import pandas as pd

from hsa.ssf.frequentist import fit_conditional_ssf


def _synthetic_choice_table(seed=7, n_strata=250, n_choices=4):
    rng = np.random.default_rng(seed)
    beta = np.array([0.7, -0.45])
    rows = []
    for stratum in range(n_strata):
        X = rng.normal(size=(n_choices, 2))
        eta = X @ beta
        eta -= eta.max()
        probability = np.exp(eta)
        probability /= probability.sum()
        chosen = int(rng.choice(n_choices, p=probability))
        for candidate in range(n_choices):
            rows.append(
                {
                    "id": "A" if stratum < n_strata // 2 else "B",
                    "stratum_id": stratum,
                    "candidate_id": candidate,
                    "used": int(candidate == chosen),
                    "x1": X[candidate, 0],
                    "x2": X[candidate, 1],
                }
            )
    return pd.DataFrame(rows)


def test_fast_conditional_ssf_matches_statsmodels_coefficients_and_se():
    df = _synthetic_choice_table()

    fast_model, fast = fit_conditional_ssf(
        df,
        predictors=["x1", "x2"],
        id_col="id",
        engine="fast",
        method="bfgs",
        maxiter=500,
    )
    sm_model, sm = fit_conditional_ssf(
        df,
        predictors=["x1", "x2"],
        id_col="id",
        engine="statsmodels",
        method="bfgs",
        maxiter=500,
    )

    np.testing.assert_allclose(
        fast.params.to_numpy(),
        sm.params.to_numpy(),
        rtol=5e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        fast.bse.to_numpy(),
        sm.bse.to_numpy(),
        rtol=2e-4,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        fast_model.hessian(fast.params),
        sm_model.hessian(sm.params),
        rtol=3e-4,
        atol=3e-5,
    )


def test_fast_score_is_zero_near_optimum():
    df = _synthetic_choice_table(n_strata=120)
    model, result = fit_conditional_ssf(
        df,
        predictors=["x1", "x2"],
        id_col="id",
        engine="fast",
    )
    assert np.linalg.norm(model.score(result.params)) < 1e-4
    assert result.converged


def test_fast_engine_rejects_variable_choice_set_size():
    df = _synthetic_choice_table(n_strata=20, n_choices=3)
    df = df.drop(df.index[-1])

    try:
        fit_conditional_ssf(
            df,
            predictors=["x1", "x2"],
            id_col="id",
            engine="fast",
        )
    except ValueError as exc:
        assert "constant choice-set size" in str(exc)
    else:
        raise AssertionError("Variable choice-set size should not be accepted.")
