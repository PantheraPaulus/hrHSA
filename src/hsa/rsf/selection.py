from __future__ import annotations

import ast
import itertools
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from hsa.rsf.model import fit_rsf
from hsa.types import FeatureSpec


def split_multiscale_name(name: str) -> tuple[str, str | None]:
    """Split a predictor name into base variable and terminal scale suffix.

    Examples
    --------
    ``ndvi_mean_100m`` -> ``("ndvi_mean", "100m")``
    ``ndvi_mean`` -> ``("ndvi_mean", None)``
    """

    match = re.match(r"^(.*)_(\d+(?:\.\d+)?m)$", str(name))
    if match:
        return match.group(1), match.group(2)
    return str(name), None


def select_predictor_columns(
    df: pd.DataFrame,
    *,
    regex: str | None = None,
    columns: Iterable[str] | None = None,
    exclude: Iterable[str] = (
        "used",
        "Timestamp",
        "time_rounded",
        "Individual_ID",
        "x",
        "y",
        "geometry",
    ),
    numeric_only: bool = True,
    case: bool = False,
) -> list[str]:
    """Select predictor columns from a sampled use-available table."""

    if columns is not None:
        selected = [col for col in columns if col in df.columns]
    elif regex is not None:
        selected = [
            col
            for col in df.columns
            if re.search(regex, str(col), flags=0 if case else re.IGNORECASE)
        ]
    else:
        selected = list(df.columns)

    excluded = set(exclude)
    selected = [col for col in selected if col not in excluded]

    if numeric_only:
        selected = [col for col in selected if pd.api.types.is_numeric_dtype(df[col])]

    return selected


def _model_row(
    *,
    name: str,
    model,
    spec: FeatureSpec,
    error: str | None = None,
) -> dict[str, Any]:
    if error is not None or model is None:
        return {
            "model": name,
            "n_params": np.nan,
            "n_obs": np.nan,
            "logLik": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "pseudo_r2": np.nan,
            "converged": False,
            "error": error,
            "linear": list(spec.linear),
            "quadratic": list(spec.quadratic),
            "categorical": list(spec.categorical),
        }

    return {
        "model": name,
        "n_params": int(model.df_model + 1),
        "n_obs": int(model.nobs),
        "logLik": float(model.llf),
        "AIC": float(model.aic),
        "BIC": float(model.bic),
        "pseudo_r2": float(getattr(model, "prsquared", np.nan)),
        "converged": bool(getattr(model, "mle_retvals", {}).get("converged", np.nan)),
        "error": None,
        "linear": list(spec.linear),
        "quadratic": list(spec.quadratic),
        "categorical": list(spec.categorical),
    }


def add_aic_weights(table: pd.DataFrame, *, aic_col: str = "AIC") -> pd.DataFrame:
    """Add delta AIC and Akaike weights to a model-comparison table."""

    out = table.copy()
    valid = out[aic_col].notna()
    out["delta_aic"] = np.nan
    out["akaike_weight"] = np.nan

    if valid.any():
        min_aic = out.loc[valid, aic_col].min()
        out.loc[valid, "delta_aic"] = out.loc[valid, aic_col] - min_aic
        weights = np.exp(-0.5 * out.loc[valid, "delta_aic"])
        out.loc[valid, "akaike_weight"] = weights / weights.sum()

    return out


def compare_rsf_specs(
    df: pd.DataFrame,
    specs: Mapping[str, FeatureSpec],
    *,
    fit_func: Callable[..., tuple] = fit_rsf,
    sort_by: str = "AIC",
    add_weights: bool = True,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Fit named RSF specifications and compare them by AIC/BIC."""

    rows: list[dict[str, Any]] = []

    for name, spec in specs.items():
        try:
            model, *_ = fit_func(df, spec, **fit_kwargs)
            rows.append(_model_row(name=name, model=model, spec=spec))
        except Exception as exc:  # intentionally records model failures
            rows.append(_model_row(name=name, model=None, spec=spec, error=repr(exc)))

    out = pd.DataFrame(rows)
    if add_weights:
        out = add_aic_weights(out)
    if sort_by in out.columns:
        out = out.sort_values(sort_by, na_position="last").reset_index(drop=True)
    return out


def compare_single_predictors(
    df: pd.DataFrame,
    predictors: Iterable[str] | None = None,
    *,
    regex: str | None = None,
    categorical: Iterable[str] | None = None,
    fit_func: Callable[..., tuple] = fit_rsf,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Fit one univariate RSF per predictor and rank by AIC."""

    if predictors is None:
        predictors = select_predictor_columns(df, regex=regex)
    predictors = [col for col in predictors if col in df.columns]
    categorical = set(categorical or [])

    specs = {
        col: FeatureSpec(
            linear=[] if col in categorical else [col],
            categorical=[col] if col in categorical else [],
            add_const=True,
        )
        for col in predictors
    }
    out = compare_rsf_specs(df, specs, fit_func=fit_func, **fit_kwargs)
    return out.rename(columns={"model": "predictor"})


def has_duplicate_base_variables(combo: Sequence[str]) -> bool:
    """Return True if a predictor combination contains repeated base variables."""

    bases = [split_multiscale_name(col)[0] for col in combo]
    return len(bases) != len(set(bases))


def _fit_one_linear_candidate(
    combo: tuple[str, ...],
    df: pd.DataFrame,
    fit_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fit_kwargs = fit_kwargs or {}
    linear = list(combo)
    spec = FeatureSpec(linear=linear, add_const=True)

    try:
        try:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=1):
                model, *_ = fit_rsf(df, spec, **fit_kwargs)
        except ImportError:
            model, *_ = fit_rsf(df, spec, **fit_kwargs)

        return {
            "Variables": linear,
            "n_vars": len(linear),
            "n_params": int(model.df_model + 1),
            "n_obs": int(model.nobs),
            "logLik": float(model.llf),
            "AIC": float(model.aic),
            "BIC": float(model.bic),
            "pseudo_r2": float(getattr(model, "prsquared", np.nan)),
            "converged": bool(getattr(model, "mle_retvals", {}).get("converged", np.nan)),
            "error": None,
        }
    except Exception as exc:  # intentionally records model failures
        return {
            "Variables": linear,
            "n_vars": len(linear),
            "n_params": np.nan,
            "n_obs": np.nan,
            "logLik": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "pseudo_r2": np.nan,
            "converged": False,
            "error": repr(exc),
        }


def evaluate_linear_candidates_up_to_k(
    df: pd.DataFrame,
    predictor_cols: Iterable[str] | None = None,
    *,
    max_k: int = 3,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 0,
    allow_multiple_scales_per_variable: bool = False,
    add_weights: bool = True,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Evaluate all linear predictor combinations up to size ``max_k``.

    Set ``n_jobs`` > 1 to evaluate combinations in parallel using joblib.
    """

    if predictor_cols is None:
        predictor_cols = select_predictor_columns(df)

    predictor_cols = [col for col in predictor_cols if col in df.columns]

    combos: list[tuple[str, ...]] = []
    for k in range(1, min(len(predictor_cols), max_k) + 1):
        for combo in itertools.combinations(predictor_cols, k):
            if not allow_multiple_scales_per_variable and has_duplicate_base_variables(combo):
                continue
            combos.append(tuple(combo))

    if n_jobs == 1:
        rows = [_fit_one_linear_candidate(combo, df, fit_kwargs) for combo in combos]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(_fit_one_linear_candidate)(combo, df, fit_kwargs) for combo in combos
        )

    out = pd.DataFrame(rows)
    if add_weights:
        out = add_aic_weights(out)
    return out.sort_values("AIC", na_position="last").reset_index(drop=True)


def evaluate_regex_candidate_family(
    df: pd.DataFrame,
    *,
    regex: str,
    max_k: int = 3,
    n_jobs: int = 1,
    allow_multiple_scales_per_variable: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Select predictors by regex and run candidate-combination AIC screening."""

    predictors = select_predictor_columns(df, regex=regex)
    return evaluate_linear_candidates_up_to_k(
        df,
        predictors,
        max_k=max_k,
        n_jobs=n_jobs,
        allow_multiple_scales_per_variable=allow_multiple_scales_per_variable,
        **kwargs,
    )


def _extract_univariate_predictor(value: Any) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected univariate model, got list of length {len(value)}: {value}")
        return str(value[0])

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, list) or len(parsed) != 1:
                raise ValueError(f"Expected univariate model, got: {value}")
            return str(parsed[0])
        return text

    raise TypeError(f"Unsupported predictor representation: {type(value)}")


def summarize_univariate_scale_selection(
    model_comparison: pd.DataFrame,
    *,
    delta_aic: float | None = 50.0,
    top_n: int | None = None,
    variable_col: str = "Variables",
    aic_col: str = "AIC",
) -> tuple[float, pd.DataFrame, list[str]]:
    """Summarize univariate AIC screening and retain best scale per base variable."""

    df = model_comparison.copy()
    if aic_col not in df.columns:
        raise KeyError(f"{aic_col!r} not found in model_comparison columns.")
    if variable_col not in df.columns:
        raise KeyError(f"{variable_col!r} not found in model_comparison columns.")

    df = df.loc[df[aic_col].notna()].copy()
    if df.empty:
        raise ValueError("No valid rows with non-missing AIC found.")

    df = df.sort_values(aic_col).reset_index(drop=True)
    df["predictor"] = df[variable_col].apply(_extract_univariate_predictor)
    df[["base_variable", "scale"]] = df["predictor"].apply(
        lambda name: pd.Series(split_multiscale_name(name))
    )

    global_best_aic = float(df[aic_col].min())

    if top_n is not None:
        retained = df.nsmallest(top_n, aic_col).copy()
        threshold_aic = float(retained[aic_col].max())
    else:
        if delta_aic is None:
            raise ValueError("Provide either delta_aic or top_n.")
        threshold_aic = global_best_aic + float(delta_aic)
        retained = df.loc[df[aic_col] <= threshold_aic].copy()

    retained["delta_to_global_best"] = retained[aic_col] - global_best_aic

    best_by_base = (
        retained.groupby("base_variable", as_index=False)[aic_col]
        .min()
        .rename(columns={aic_col: "best_aic_within_base"})
    )

    retained = retained.merge(best_by_base, on="base_variable", how="left")
    retained["delta_to_best_scale_within_base"] = (
        retained[aic_col] - retained["best_aic_within_base"]
    )
    retained["is_best_scale_for_base"] = retained["delta_to_best_scale_within_base"] == 0

    keep_cols = [
        "predictor",
        "base_variable",
        "best_aic_within_base",
        "scale",
        aic_col,
        "delta_to_global_best",
        "delta_to_best_scale_within_base",
        "is_best_scale_for_base",
    ]
    summary = (
        retained[keep_cols]
        .sort_values(["best_aic_within_base", "base_variable", aic_col])
        .reset_index(drop=True)
    )

    best_predictors = (
        summary.loc[summary["is_best_scale_for_base"], "predictor"]
        .drop_duplicates()
        .tolist()
    )

    return threshold_aic, summary, best_predictors


def select_best_scale_per_predictor(screening_table: pd.DataFrame) -> pd.DataFrame:
    """Keep the minimum-AIC row for each predictor in a screening table."""

    table = screening_table.copy()
    if "status" in table.columns:
        table = table.loc[table["status"] == "ok"].copy()
    table = table.loc[table["AIC"].notna()].copy()
    if table.empty:
        return table

    group_col = "predictor" if "predictor" in table.columns else "base_variable"
    best_idx = table.groupby(group_col)["AIC"].idxmin()
    sort_cols = [col for col in ("group", "AIC") if col in table.columns]
    return table.loc[best_idx].sort_values(sort_cols or ["AIC"]).reset_index(drop=True)


def variable_frequency_summary(
    model_comparison: pd.DataFrame,
    *,
    variable_col: str = "Variables",
    top_n: int | None = 50,
    delta_aic: float | None = None,
    aic_col: str = "AIC",
) -> pd.DataFrame:
    """Count how often predictors occur among top-ranked candidate models."""

    df = model_comparison.loc[model_comparison[aic_col].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["variable", "n_models", "best_aic", "mean_delta_aic"])

    df = df.sort_values(aic_col).reset_index(drop=True)
    best_aic = float(df[aic_col].min())

    if top_n is not None:
        df = df.head(top_n).copy()
    if delta_aic is not None:
        df = df.loc[df[aic_col] <= best_aic + float(delta_aic)].copy()

    rows = []
    for _, row in df.iterrows():
        variables = row[variable_col]
        if isinstance(variables, str):
            if variables.strip().startswith("["):
                variables = ast.literal_eval(variables)
            else:
                variables = [variables]
        for variable in variables:
            rows.append({"variable": variable, "AIC": row[aic_col], "delta_aic": row[aic_col] - best_aic})

    if not rows:
        return pd.DataFrame(columns=["variable", "n_models", "best_aic", "mean_delta_aic"])

    out = pd.DataFrame(rows)
    return (
        out.groupby("variable", as_index=False)
        .agg(
            n_models=("variable", "size"),
            best_aic=("AIC", "min"),
            mean_delta_aic=("delta_aic", "mean"),
        )
        .sort_values(["n_models", "best_aic"], ascending=[False, True])
        .reset_index(drop=True)
    )
