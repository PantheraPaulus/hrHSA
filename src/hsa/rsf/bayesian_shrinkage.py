"""Shrinkage priors and collinearity diagnostics for Bayesian RSFs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


_ALLOWED_BETA_PRIORS = {"normal", "regularized_horseshoe"}


def _normalize_beta_prior(beta_prior: str) -> str:
    prior = str(beta_prior).strip().lower().replace("-", "_")
    if prior not in _ALLOWED_BETA_PRIORS:
        raise ValueError(
            "beta_prior must be 'normal' or 'regularized_horseshoe'."
        )
    return prior


def _resolve_horseshoe_config(
    predictors: Sequence[str],
    used_count: np.ndarray,
    shrinkage: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Validate regularized-horseshoe settings and derive its global scale."""
    predictors = list(predictors)
    p = len(predictors)
    if p < 2:
        raise ValueError(
            "regularized_horseshoe requires at least two predictors."
        )

    config = {} if shrinkage is None else dict(shrinkage)
    allowed = {
        "expected_nonzero",
        "global_scale",
        "effective_n",
        "global_df",
        "local_df",
        "slab_scale",
        "slab_df",
    }
    unknown = sorted(set(config).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown shrinkage settings: {unknown}")

    expected_nonzero = config.get("expected_nonzero")
    global_scale = config.get("global_scale")
    if expected_nonzero is None and global_scale is None:
        raise ValueError(
            "regularized_horseshoe requires shrinkage['expected_nonzero'] "
            "or shrinkage['global_scale']."
        )
    if expected_nonzero is not None and global_scale is not None:
        raise ValueError(
            "Specify only one of shrinkage['expected_nonzero'] and "
            "shrinkage['global_scale']."
        )
    if global_scale is not None and "effective_n" in config:
        raise ValueError(
            "shrinkage['effective_n'] is only used when deriving the global "
            "scale from shrinkage['expected_nonzero']."
        )

    if expected_nonzero is not None:
        expected_nonzero = float(expected_nonzero)
        if not 0 < expected_nonzero < p:
            raise ValueError(
                "shrinkage['expected_nonzero'] must be > 0 and smaller than "
                f"the number of predictors ({p})."
            )

        effective_n = config.get("effective_n")
        if effective_n is None:
            # Availability points are user-controlled pseudo-absences in an RSF.
            # Using all Binomial trials would make the prior depend on the
            # arbitrary sampling factor, so the default information scale is
            # the number of used locations represented by the fitted data.
            effective_n = float(
                np.asarray(used_count, dtype=np.int64).sum()
            )
        else:
            effective_n = float(effective_n)

        if not np.isfinite(effective_n) or effective_n <= 0:
            raise ValueError("shrinkage['effective_n'] must be positive.")

        global_scale = (
            expected_nonzero
            / (p - expected_nonzero)
            / np.sqrt(effective_n)
        )
    else:
        global_scale = float(global_scale)
        effective_n = np.nan
        if not np.isfinite(global_scale) or global_scale <= 0:
            raise ValueError("shrinkage['global_scale'] must be positive.")

    resolved = {
        "global_scale": float(global_scale),
        "global_df": float(config.get("global_df", 2.0)),
        "local_df": float(config.get("local_df", 5.0)),
        "slab_scale": float(config.get("slab_scale", 2.0)),
        "slab_df": float(config.get("slab_df", 4.0)),
        "effective_n": float(effective_n),
    }
    if expected_nonzero is not None:
        resolved["expected_nonzero"] = expected_nonzero

    for key in ("global_df", "local_df", "slab_scale", "slab_df"):
        if not np.isfinite(resolved[key]) or resolved[key] <= 0:
            raise ValueError(f"shrinkage[{key!r}] must be positive.")

    return resolved


def build_population_beta_prior(
    *,
    predictors: Sequence[str],
    used_count: np.ndarray,
    beta_prior: str = "normal",
    beta_prior_sigma: float = 1.0,
    shrinkage: Mapping[str, Any] | None = None,
):
    """Construct population-level beta coefficients for a PyMC RSF.

    ``normal`` keeps the existing independent Normal prior. The
    ``regularized_horseshoe`` option uses a global-local shrinkage prior with a
    finite slab. Its global scale can be specified directly or derived from a
    prior expected number of non-zero coefficients.

    When the global scale is derived, the default effective sample size is the
    number of used locations rather than used + available trials. This keeps
    prior shrinkage invariant to the arbitrary availability sampling factor.
    """
    try:
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyMC is required; install hsa[bayesian].") from exc

    prior = _normalize_beta_prior(beta_prior)
    predictors = list(predictors)

    if prior == "normal":
        if shrinkage:
            raise ValueError(
                "shrinkage settings are only valid when "
                "beta_prior='regularized_horseshoe'."
            )
        beta_prior_sigma = float(beta_prior_sigma)
        if not np.isfinite(beta_prior_sigma) or beta_prior_sigma <= 0:
            raise ValueError("beta_prior_sigma must be positive.")
        return pm.Normal(
            "beta",
            mu=0.0,
            sigma=beta_prior_sigma,
            dims="predictor",
        )

    config = _resolve_horseshoe_config(
        predictors,
        used_count,
        shrinkage,
    )

    hs_tau = pm.HalfStudentT(
        "hs_tau",
        nu=config["global_df"],
        sigma=config["global_scale"],
    )
    hs_lambda = pm.HalfStudentT(
        "hs_lambda",
        nu=config["local_df"],
        sigma=1.0,
        dims="predictor",
    )
    hs_c2 = pm.InverseGamma(
        "hs_c2",
        alpha=config["slab_df"] / 2.0,
        beta=config["slab_df"] * config["slab_scale"] ** 2 / 2.0,
    )
    hs_beta_raw = pm.Normal(
        "hs_beta_raw",
        mu=0.0,
        sigma=1.0,
        dims="predictor",
    )

    hs_lambda_tilde = pm.Deterministic(
        "hs_lambda_tilde",
        hs_lambda
        * pt.sqrt(
            hs_c2
            / (
                hs_c2
                + hs_tau**2 * hs_lambda**2
            )
        ),
        dims="predictor",
    )
    beta_prior_scale = pm.Deterministic(
        "beta_prior_scale",
        hs_tau * hs_lambda_tilde,
        dims="predictor",
    )
    return pm.Deterministic(
        "beta",
        hs_beta_raw * beta_prior_scale,
        dims="predictor",
    )


def posterior_beta_correlation(
    idata,
    *,
    predictors: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the pairwise posterior correlation matrix of population betas."""
    if "beta" not in idata.posterior:
        raise KeyError("InferenceData posterior does not contain 'beta'.")

    beta = idata.posterior["beta"]
    if "predictor" not in beta.dims:
        raise ValueError("Posterior 'beta' must have a 'predictor' dimension.")

    available = [str(value) for value in beta["predictor"].values]
    selected = available if predictors is None else list(predictors)
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(f"Predictors not present in posterior beta: {missing}")

    beta = beta.sel(predictor=selected)
    if "sample" in beta.dims:
        stacked = beta.transpose("sample", "predictor")
    elif {"chain", "draw"}.issubset(beta.dims):
        stacked = beta.stack(sample=("chain", "draw")).transpose(
            "sample",
            "predictor",
        )
    else:
        raise ValueError(
            "Posterior beta must contain 'sample' or ('chain', 'draw') dimensions."
        )

    values = np.asarray(stacked.values, dtype=float)
    corr = np.corrcoef(values, rowvar=False)
    corr = np.atleast_2d(corr)
    return pd.DataFrame(corr, index=selected, columns=selected)


def predictor_correlation(
    bayes_data: Mapping[str, Any],
    *,
    predictors: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return trial-weighted predictor correlations from the fitted design matrix."""
    available = list(bayes_data["predictors"])
    selected = available if predictors is None else list(predictors)
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(f"Predictors not present in bayes_data: {missing}")

    predictor_idx = [available.index(predictor) for predictor in selected]
    X = np.asarray(
        bayes_data["X"][:, predictor_idx],
        dtype=float,
    )
    weights = np.asarray(
        bayes_data["n_trials"],
        dtype=float,
    )
    if len(weights) != X.shape[0]:
        raise ValueError("n_trials and X must have the same observation count.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("n_trials must contain finite non-negative weights.")
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("n_trials must sum to a positive value.")

    mean = np.average(X, axis=0, weights=weights)
    centered = X - mean
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    sd = np.sqrt(np.diag(covariance))
    denom = np.outer(sd, sd)
    corr = np.divide(
        covariance,
        denom,
        out=np.full_like(covariance, np.nan, dtype=float),
        where=denom > 0,
    )
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=selected, columns=selected)


def collinearity_diagnostics(
    idata,
    bayes_data: Mapping[str, Any],
    *,
    predictors: Sequence[str] | None = None,
    min_abs_predictor_corr: float = 0.5,
) -> pd.DataFrame:
    """Compare predictor and posterior-beta correlations for predictor pairs.

    Strong predictor correlation together with an oppositely signed posterior
    coefficient correlation is a useful indicator that the model is trading
    effect size between predictors. It is an identifiability diagnostic, not a
    causal-variable-selection result.
    """
    if not 0 <= min_abs_predictor_corr <= 1:
        raise ValueError("min_abs_predictor_corr must be between 0 and 1.")

    x_corr = predictor_correlation(
        bayes_data,
        predictors=predictors,
    )
    beta_corr = posterior_beta_correlation(
        idata,
        predictors=x_corr.columns.tolist(),
    )

    rows: list[dict[str, Any]] = []
    names = x_corr.columns.tolist()
    for i, predictor_a in enumerate(names):
        for predictor_b in names[i + 1 :]:
            predictor_corr = float(x_corr.loc[predictor_a, predictor_b])
            posterior_corr = float(beta_corr.loc[predictor_a, predictor_b])
            if (
                np.isfinite(predictor_corr)
                and abs(predictor_corr) < min_abs_predictor_corr
            ):
                continue

            competing = (
                np.isfinite(predictor_corr)
                and np.isfinite(posterior_corr)
                and predictor_corr * posterior_corr < 0
            )
            rows.append(
                {
                    "predictor_a": predictor_a,
                    "predictor_b": predictor_b,
                    "predictor_corr": predictor_corr,
                    "posterior_beta_corr": posterior_corr,
                    "abs_predictor_corr": abs(predictor_corr),
                    "abs_posterior_beta_corr": abs(posterior_corr),
                    "competing_posterior": bool(competing),
                }
            )

    columns = [
        "predictor_a",
        "predictor_b",
        "predictor_corr",
        "posterior_beta_corr",
        "abs_predictor_corr",
        "abs_posterior_beta_corr",
        "competing_posterior",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["abs_predictor_corr", "abs_posterior_beta_corr"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def regularized_horseshoe_summary(
    idata,
    *,
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Summarize coefficient escape from a fitted regularized horseshoe prior.

    ``beta_prior_scale`` and the local-scale columns describe posterior
    regularization strength. They are not posterior inclusion probabilities.
    """
    if not 0 < ci_prob < 1:
        raise ValueError("ci_prob must be between 0 and 1.")

    posterior = idata.posterior
    required = {
        "beta",
        "hs_tau",
        "hs_lambda",
        "hs_lambda_tilde",
        "beta_prior_scale",
    }
    missing = sorted(required.difference(posterior.data_vars))
    if missing:
        raise ValueError(
            "InferenceData does not contain a regularized-horseshoe posterior; "
            f"missing variables: {missing}"
        )

    beta = posterior["beta"]
    predictors = [str(value) for value in beta["predictor"].values]
    alpha = (1.0 - ci_prob) / 2.0
    tau = np.asarray(posterior["hs_tau"].values, dtype=float).ravel()
    tau_median = float(np.median(tau))

    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        beta_values = np.asarray(
            beta.sel(predictor=predictor).values,
            dtype=float,
        ).ravel()
        local = np.asarray(
            posterior["hs_lambda"].sel(predictor=predictor).values,
            dtype=float,
        ).ravel()
        local_tilde = np.asarray(
            posterior["hs_lambda_tilde"].sel(predictor=predictor).values,
            dtype=float,
        ).ravel()
        prior_scale = np.asarray(
            posterior["beta_prior_scale"].sel(predictor=predictor).values,
            dtype=float,
        ).ravel()

        rows.append(
            {
                "predictor": predictor,
                "beta_mean": float(np.mean(beta_values)),
                "beta_median": float(np.median(beta_values)),
                "beta_lower": float(np.quantile(beta_values, alpha)),
                "beta_upper": float(np.quantile(beta_values, 1.0 - alpha)),
                "p_beta_gt_zero": float(np.mean(beta_values > 0)),
                "tau_median": tau_median,
                "local_scale_median": float(np.median(local)),
                "regularized_local_scale_median": float(np.median(local_tilde)),
                "beta_prior_scale_median": float(np.median(prior_scale)),
                "beta_prior_scale_lower": float(np.quantile(prior_scale, alpha)),
                "beta_prior_scale_upper": float(
                    np.quantile(prior_scale, 1.0 - alpha)
                ),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "build_population_beta_prior",
    "posterior_beta_correlation",
    "predictor_correlation",
    "collinearity_diagnostics",
    "regularized_horseshoe_summary",
]
