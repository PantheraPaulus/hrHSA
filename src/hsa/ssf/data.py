"""Canonical SSF design-table preparation and choice-array conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SSFChoiceArrays:
    """Dense fixed-size choice-set representation used by SSF estimators."""

    X: np.ndarray
    chosen: np.ndarray
    individual_idx: np.ndarray
    individuals: tuple[Any, ...]
    strata: pd.DataFrame
    predictors: tuple[str, ...]
    n_choices: int
    offset: np.ndarray | None = None
    offset_name: str | None = None


def complete_ssf_strata(
    df: pd.DataFrame,
    *,
    predictors: Sequence[str],
    id_col: str,
    stratum_col: str = "stratum_id",
    used_col: str = "used",
    expected_n_choices: int | None = None,
) -> pd.DataFrame:
    """Drop incomplete *whole strata* and retain structurally valid choice sets."""
    required = [id_col, stratum_col, used_col, *predictors]
    missing = [column for column in required if column not in df]
    if missing:
        raise KeyError(f"Missing SSF columns: {missing}")

    d = df.replace([np.inf, -np.inf], np.nan).copy()
    row_complete = d[required].notna().all(axis=1)
    complete = row_complete.groupby(
        [d[id_col], d[stratum_col]],
        sort=False,
    ).transform("all")
    d = d.loc[complete].copy()

    counts = d.groupby([id_col, stratum_col], sort=False)[used_col].agg(
        n_choices="size",
        n_used="sum",
    )
    valid = counts["n_used"].eq(1)
    if expected_n_choices is not None:
        valid &= counts["n_choices"].eq(expected_n_choices)

    valid_index = counts.index[valid]
    keys = pd.MultiIndex.from_frame(d[[id_col, stratum_col]])
    d = d.loc[keys.isin(valid_index)].copy()
    if d.empty:
        raise ValueError("No complete valid SSF strata remain.")
    return d


def fit_ssf_scaling(
    df: pd.DataFrame,
    predictors: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Fit one global continuous-predictor scaler to training choices only."""
    scaling: dict[str, dict[str, float]] = {}
    for predictor in predictors:
        if predictor not in df:
            raise KeyError(f"{predictor!r} not found in SSF table.")
        values = (
            pd.to_numeric(df[predictor], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        mean = float(values.mean())
        sd = float(values.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"{predictor!r} has invalid SD: {sd}")
        scaling[predictor] = {"mean": mean, "sd": sd}
    return scaling


def apply_ssf_scaling(
    df: pd.DataFrame,
    predictors: Sequence[str],
    scaling: Mapping[str, Mapping[str, float]],
    *,
    suffix: str = "_z",
) -> pd.DataFrame:
    """Apply pre-fitted scaling without leaking held-out data."""
    out = df.copy()
    for predictor in predictors:
        if predictor not in scaling:
            raise KeyError(f"Scaling metadata missing for {predictor!r}.")
        mean = float(scaling[predictor]["mean"])
        sd = float(scaling[predictor]["sd"])
        out[f"{predictor}{suffix}"] = (
            out[predictor] - mean
        ) / sd
    return out


def build_ssf_choice_arrays(
    df: pd.DataFrame,
    *,
    id_col: str,
    predictors: Sequence[str],
    stratum_col: str = "stratum_id",
    candidate_col: str = "candidate_id",
    used_col: str = "used",
    offset_col: str | None = None,
    dtype: str | np.dtype = "float32",
) -> SSFChoiceArrays:
    """Convert a complete fixed-size SSF table to dense choice tensors.

    ``float32`` remains the default because it is well suited to the Bayesian
    PyMC tensor. Frequentist optimizers can request ``dtype='float64'`` to retain
    full numerical precision while using the same canonical array builder.
    """
    predictors = tuple(predictors)
    required = [
        id_col,
        stratum_col,
        candidate_col,
        used_col,
        *predictors,
    ]

    if offset_col is not None:
        required.append(offset_col)

    missing = [column for column in required if column not in df]
    if missing:
        raise KeyError(f"Missing SSF columns: {missing}")

    d = df.sort_values(
        [id_col, stratum_col, candidate_col]
    ).copy()
    grouped = d.groupby([id_col, stratum_col], sort=False)
    check = grouped[used_col].agg(
        n_choices="size",
        n_used="sum",
    )
    if not check["n_used"].eq(1).all():
        raise ValueError("Every SSF stratum must contain exactly one used choice.")

    candidate_unique = grouped[candidate_col].nunique()
    if not candidate_unique.eq(check["n_choices"]).all():
        raise ValueError("candidate_id must be unique within every SSF stratum.")

    sizes = check["n_choices"].unique()
    if len(sizes) != 1:
        raise ValueError(
            "SSF choice arrays require constant choice-set size."
        )
    n_choices = int(sizes[0])

    individuals = tuple(
        pd.Index(d[id_col].drop_duplicates()).tolist()
    )
    lookup = {
        animal: i
        for i, animal in enumerate(individuals)
    }
    strata = (
        d.drop_duplicates([id_col, stratum_col])[[id_col, stratum_col]]
        .reset_index(drop=True)
    )
    strata["individual_idx"] = (
        strata[id_col].map(lookup).astype("int32")
    )
    n_strata = len(strata)

    X = (
        d[list(predictors)]
        .to_numpy(dtype=dtype)
        .reshape(n_strata, n_choices, len(predictors))
    )
    if not np.isfinite(X).all():
        raise ValueError("SSF design tensor contains non-finite predictor values.")

    offset = None

    if offset_col is not None:

        offset = (
            d[offset_col]
            .to_numpy(dtype=dtype)
            .reshape(n_strata, n_choices)
        )

        if not np.isfinite(offset).all():
            raise ValueError(
                "SSF offset tensor contains non-finite values."
            )

    used = d[used_col].to_numpy().reshape(n_strata, n_choices)
    chosen = used.argmax(axis=1).astype("int32")
    individual_idx = strata["individual_idx"].to_numpy(dtype="int32")

    return SSFChoiceArrays(
        X=X,
        chosen=chosen,
        individual_idx=individual_idx,
        individuals=individuals,
        strata=strata,
        predictors=predictors,
        n_choices=n_choices,
        offset=offset,
        offset_name=offset_col,
    )

def stable_softmax(eta: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    eta = np.asarray(eta, dtype=float)
    shifted = eta - np.max(eta, axis=axis, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=axis, keepdims=True)


def score_choice_probabilities(
    probabilities: np.ndarray,
    chosen: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series]:
    """Score chosen alternatives against a uniform-choice null model.

    Tied alternatives receive their average rank, matching
    ``pandas.Series.rank(method='average', ascending=False)``.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    chosen = np.asarray(chosen, dtype=int)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (stratum, choice).")

    n_strata, n_choices = probabilities.shape
    if chosen.shape != (n_strata,):
        raise ValueError("chosen must contain one candidate index per stratum.")
    if np.any(chosen < 0) or np.any(chosen >= n_choices):
        raise ValueError("chosen contains candidate indices outside the choice set.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Choice probabilities must be finite and non-negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Choice probabilities must sum to one within each stratum.")

    p_chosen = probabilities[np.arange(n_strata), chosen]
    p_chosen = np.clip(p_chosen, np.finfo(float).tiny, 1.0)

    greater = (probabilities > p_chosen[:, None]).sum(axis=1)
    tied = (probabilities == p_chosen[:, None]).sum(axis=1)
    ranks = 1.0 + greater + 0.5 * (tied - 1)

    nll = -np.log(p_chosen)
    null_nll = float(np.log(n_choices))
    gain = null_nll - nll
    rank_percentile = 1.0 - (
        (ranks - 1.0) / max(n_choices - 1, 1)
    )

    per_stratum = pd.DataFrame(
        {
            "chosen_probability": p_chosen,
            "choice_rank": ranks,
            "rank_percentile": rank_percentile,
            "negative_log_score": nll,
            "null_negative_log_score": null_nll,
            "log_score_gain": gain,
        }
    )
    summary = pd.Series(
        {
            "n_strata": n_strata,
            "n_choices": n_choices,
            "mean_chosen_probability": float(p_chosen.mean()),
            "median_chosen_probability": float(np.median(p_chosen)),
            "top_1": float(np.mean(ranks == 1)),
            "top_5": float(np.mean(ranks <= min(5, n_choices))),
            "top_half": float(np.mean(ranks <= np.ceil(n_choices / 2))),
            "mean_rank_percentile": float(rank_percentile.mean()),
            "mean_negative_log_score": float(nll.mean()),
            "null_negative_log_score": null_nll,
            "mean_log_score_gain": float(gain.mean()),
            "median_log_score_gain": float(np.median(gain)),
            "fraction_gain_positive": float(np.mean(gain > 0)),
            "total_elpd_gain": float(gain.sum()),
            "predictive_advantage": float(np.exp(gain.mean())),
        }
    )
    return per_stratum, summary


__all__ = [
    "SSFChoiceArrays",
    "complete_ssf_strata",
    "fit_ssf_scaling",
    "apply_ssf_scaling",
    "build_ssf_choice_arrays",
    "score_choice_probabilities",
    "stable_softmax",
]
