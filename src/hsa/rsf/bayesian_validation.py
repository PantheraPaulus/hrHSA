"""Posterior Boyce validation for hierarchical Bayesian RSFs.

Two complementary uncertainty modes are implemented:

``bootstrap``
    A temporally blocked bootstrap of the held-out trajectory. Whole temporal
    blocks are sampled with replacement and never truncated. One posterior
    coefficient draw is paired with each bootstrap replicate, so the resulting
    distribution combines parameter uncertainty with finite validation-sample
    uncertainty.

``temporal_contiguous``
    The real held-out chronology is split into contiguous temporal periods.
    Every period is evaluated across all posterior draws. This is an empirical
    diagnostic of temporal non-stationarity, not a bootstrap/credible interval.

Random temporal folds are deliberately not exposed: once the bootstrap itself
is temporally blocked they provide largely redundant information.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from hsa.rsf.validation_utils import (
    assign_contiguous_temporal_folds,
    boyce_curve_from_ranks,
    fixed_duration_block_labels,
)
from hsa.sampling import (
    get_availability_domain,
    sample_available_points,
    sample_raster_stack,
)


def _sample_frame(*args, **kwargs) -> pd.DataFrame:
    """Return the dataframe across old/new sample_raster_stack return contracts."""
    sampled = sample_raster_stack(*args, **kwargs)
    return sampled[0] if isinstance(sampled, tuple) else sampled


def _extract_population_beta_draws(
    idata,
    predictors: Sequence[str],
    *,
    n_draws: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Return population beta draws as (sample, predictor)."""
    predictors = list(predictors)
    beta = idata.posterior["beta"]
    if "predictor" not in beta.dims:
        raise ValueError(
            "Posterior variable 'beta' must have a 'predictor' dimension."
        )

    available = [str(v) for v in beta["predictor"].values]
    missing = sorted(set(predictors).difference(available))
    if missing:
        raise ValueError(f"Predictors not present in posterior beta: {missing}")

    beta = (
        beta.sel(predictor=predictors)
        .stack(sample=("chain", "draw"))
        .transpose("sample", "predictor")
    )

    if n_draws is not None and beta.sizes["sample"] > n_draws:
        rng = np.random.default_rng(seed)
        idx = np.sort(
            rng.choice(
                beta.sizes["sample"],
                size=n_draws,
                replace=False,
            )
        )
        beta = beta.isel(sample=idx)

    return np.asarray(beta.values, dtype=np.float32)


def _standardize_validation_frame(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    meta: Mapping,
) -> np.ndarray:
    """Standardize sampled raster predictors with training-set metadata."""
    columns = []
    for predictor in predictors:
        if predictor not in frame.columns:
            raise KeyError(
                f"Predictor {predictor!r} missing from sampled validation data."
            )
        if predictor not in meta:
            raise KeyError(
                f"Training metadata missing for predictor {predictor!r}."
            )

        mean = float(meta[predictor]["mean"])
        sd = float(meta[predictor]["sd"])
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                f"Invalid training standard deviation for {predictor!r}: {sd}"
            )
        columns.append(
            (frame[predictor].to_numpy(dtype=np.float32) - mean) / sd
        )

    return np.column_stack(columns).astype(np.float32, copy=False)


def _select_predictor_env(env, predictors: Sequence[str]):
    """Return a lazy raster view containing only requested predictor bands."""
    predictors = list(dict.fromkeys(predictors))
    if "band" not in env.dims:
        raise ValueError("env must contain a 'band' dimension.")

    available = {str(value) for value in env["band"].values}
    missing = [predictor for predictor in predictors if predictor not in available]
    if missing:
        raise ValueError(
            "Predictors required for Bayesian validation are missing from env: "
            f"{missing}"
        )
    return env.sel(band=predictors)


def prepare_bayesian_boyce_scores(
    used: gpd.GeoDataFrame,
    env,
    idata,
    meta: Mapping,
    *,
    predictors: Sequence[str],
    domain: gpd.GeoDataFrame | None = None,
    domain_quantile: float = 0.95,
    n_background: int = 100_000,
    n_draws: int | None = 500,
    seed: int = 42,
    exponentiate: bool = False,
) -> dict[str, Any]:
    """Prepare posterior scores at held-out used and fixed available points.

    Available coordinates are sampled exactly once and reused for every
    posterior draw. Only the model predictor bands are sampled from ``env``.
    By default the linear predictor is returned because Boyce is rank based and
    therefore does not require exponentiation.
    """
    if used.crs is None:
        raise ValueError("used.crs is None; set a CRS before Boyce validation.")

    predictors = list(dict.fromkeys(predictors))
    if not predictors:
        raise ValueError("At least one predictor is required.")
    if n_background <= 0:
        raise ValueError("n_background must be positive.")

    env_model = _select_predictor_env(env, predictors)

    dom = (
        get_availability_domain(
            used,
            estimator="mcp",
            quantile=domain_quantile,
        )
        if domain is None
        else domain.copy()
    )
    if dom.crs != used.crs:
        dom = dom.to_crs(used.crs)

    available_points = sample_available_points(
        dom,
        n_background,
        seed=seed,
    )

    used_points = used.copy()
    used_points["used"] = True

    used_data = _sample_frame(
        used_points,
        env_model,
        bands=predictors,
    ).replace([np.inf, -np.inf], np.nan)
    available_data = _sample_frame(
        available_points,
        env_model,
        bands=predictors,
    ).replace([np.inf, -np.inf], np.nan)

    used_valid = used_data[predictors].notna().all(axis=1).to_numpy()
    available_valid = (
        available_data[predictors].notna().all(axis=1).to_numpy()
    )

    used_data = used_data.loc[used_valid].reset_index(drop=True)
    available_data = available_data.loc[available_valid].reset_index(drop=True)
    used_points = used_points.loc[used_valid].reset_index(drop=True)
    available_points = available_points.loc[available_valid].reset_index(drop=True)

    if used_data.empty:
        raise ValueError(
            "No held-out used locations have complete predictor values."
        )
    if available_data.empty:
        raise ValueError(
            "No available locations have complete predictor values."
        )

    X_used = _standardize_validation_frame(
        used_data,
        predictors,
        meta,
    )
    X_available = _standardize_validation_frame(
        available_data,
        predictors,
        meta,
    )
    beta_draws = _extract_population_beta_draws(
        idata,
        predictors,
        n_draws=n_draws,
        seed=seed,
    )

    used_scores = (beta_draws @ X_used.T).astype(np.float32, copy=False)
    available_scores = (beta_draws @ X_available.T).astype(
        np.float32,
        copy=False,
    )

    if exponentiate:
        used_scores = np.exp(used_scores).astype(np.float32, copy=False)
        available_scores = np.exp(available_scores).astype(
            np.float32,
            copy=False,
        )

    return {
        "used_scores": used_scores,
        "available_scores": available_scores,
        "used_data": used_data,
        "available_data": available_data,
        "used_points": used_points,
        "available_points": available_points,
        "beta_draws": beta_draws,
        "domain": dom,
        "predictors": predictors,
        "env": env_model,
        "exponentiate": exponentiate,
    }


def _used_percentile_ranks(
    used_scores: np.ndarray,
    available_scores: np.ndarray,
) -> np.ndarray:
    """Convert used scores to percentile ranks in the available distribution."""
    used_scores = np.asarray(used_scores, dtype=float)
    available_scores = np.asarray(available_scores, dtype=float)

    if used_scores.ndim != 2 or available_scores.ndim != 2:
        raise ValueError(
            "Score arrays must both be 2-D: (posterior_draw, location)."
        )
    if used_scores.shape[0] != available_scores.shape[0]:
        raise ValueError(
            "Used and available scores must have equal posterior draw counts."
        )

    ranks = np.full(used_scores.shape, np.nan, dtype=np.float32)
    for sample in range(used_scores.shape[0]):
        available = available_scores[sample]
        available = np.sort(available[np.isfinite(available)])
        if len(available) == 0:
            continue

        used = used_scores[sample]
        valid = np.isfinite(used)
        ranks[sample, valid] = (
            np.searchsorted(
                available,
                used[valid],
                side="right",
            )
            / len(available)
        )

    return ranks


def bayesian_boyce_quantile_scores(
    used_scores: np.ndarray,
    available_scores: np.ndarray,
    *,
    n_bins: int = 20,
    ci_prob: float = 0.95,
) -> dict[str, Any]:
    """Propagate posterior coefficient draws through a quantile-bin Boyce curve."""
    ranks = _used_percentile_ranks(used_scores, available_scores)
    n_draws = ranks.shape[0]
    pe_draws = np.full((n_draws, n_bins), np.nan, dtype=float)
    boyce_draws = np.full(n_draws, np.nan, dtype=float)
    rank = None

    for sample in range(n_draws):
        rank, pe_draws[sample], boyce_draws[sample] = boyce_curve_from_ranks(
            ranks[sample],
            n_bins=n_bins,
        )

    alpha = (1.0 - ci_prob) / 2.0
    curve_summary = pd.DataFrame(
        {
            "rank": rank,
            "pe_mean": np.nanmean(pe_draws, axis=0),
            "pe_median": np.nanmedian(pe_draws, axis=0),
            "pe_lower": np.nanquantile(pe_draws, alpha, axis=0),
            "pe_upper": np.nanquantile(pe_draws, 1.0 - alpha, axis=0),
            "p_pe_gt_one": np.nanmean(pe_draws > 1.0, axis=0),
        }
    )

    curve_draws = pd.DataFrame(
        {
            "sample": np.repeat(np.arange(n_draws), n_bins),
            "rank": np.tile(rank, n_draws),
            "pe_ratio": pe_draws.ravel(),
        }
    )

    finite = boyce_draws[np.isfinite(boyce_draws)]
    if len(finite):
        boyce_summary = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "lower": float(np.quantile(finite, alpha)),
            "upper": float(np.quantile(finite, 1.0 - alpha)),
            "p_gt_zero": float(np.mean(finite > 0)),
            "n_samples": int(len(finite)),
        }
    else:
        boyce_summary = {
            "mean": np.nan,
            "median": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "p_gt_zero": np.nan,
            "n_samples": 0,
        }

    return {
        "boyce_draws": boyce_draws,
        "boyce_summary": boyce_summary,
        "curve_draws": curve_draws,
        "curve_summary": curve_summary,
        "pe_draws": pe_draws,
        "rank": rank,
    }


def plot_bayesian_boyce(
    result: Mapping,
    *,
    show_draws: int = 30,
    ci_prob: float = 0.95,
    figsize: tuple[float, float] = (12, 5),
    random_seed: int = 42,
):
    """Plot posterior P/E curves and the scalar posterior Boyce distribution."""
    import matplotlib.pyplot as plt

    curve = result["curve_summary"]
    curve_draws = result["curve_draws"]
    boyce_draws = np.asarray(result["boyce_draws"], dtype=float)
    boyce_draws = boyce_draws[np.isfinite(boyce_draws)]

    fig, (ax_curve, ax_boyce) = plt.subplots(1, 2, figsize=figsize)

    samples = curve_draws["sample"].unique()
    if show_draws and len(samples):
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(
            samples,
            size=min(show_draws, len(samples)),
            replace=False,
        )
        for sample in selected:
            data = curve_draws.loc[curve_draws["sample"] == sample]
            ax_curve.plot(
                data["rank"],
                data["pe_ratio"],
                linewidth=0.7,
                alpha=0.08,
            )

    ax_curve.fill_between(
        curve["rank"].to_numpy(),
        curve["pe_lower"].to_numpy(),
        curve["pe_upper"].to_numpy(),
        alpha=0.18,
        label=f"{ci_prob:.0%} posterior interval",
    )
    ax_curve.plot(
        curve["rank"],
        curve["pe_median"],
        linewidth=2.2,
        label="Posterior median",
    )
    ax_curve.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="Use = availability",
    )
    ax_curve.set_xlabel("Predicted selection rank in available habitat")
    ax_curve.set_ylabel("Predicted / expected use (P/E)")
    ax_curve.set_xlim(0, 1)
    ax_curve.legend(frameon=False)

    if len(boyce_draws):
        ax_boyce.hist(
            boyce_draws,
            bins=min(30, max(1, len(np.unique(boyce_draws)))),
            density=True,
            alpha=0.35,
        )
        ax_boyce.axvline(np.median(boyce_draws), linewidth=2)
    ax_boyce.axvline(0, linestyle="--", linewidth=1)
    ax_boyce.set_xlim(-1, 1)
    ax_boyce.set_xlabel("Boyce index")
    ax_boyce.set_ylabel("Density")

    fig.tight_layout()
    return fig, (ax_curve, ax_boyce)


def _evaluate_posterior_subset(
    used_ranks: np.ndarray,
    indices: np.ndarray,
    *,
    n_bins: int,
    ci_prob: float,
) -> dict[str, Any]:
    """Evaluate one subset of a held-out trajectory across all posterior draws."""
    indices = np.asarray(indices, dtype=int)
    n_draws = used_ranks.shape[0]
    pe_draws = np.full((n_draws, n_bins), np.nan, dtype=float)
    boyce_draws = np.full(n_draws, np.nan, dtype=float)
    rank = None

    for sample in range(n_draws):
        rank, pe_draws[sample], boyce_draws[sample] = boyce_curve_from_ranks(
            used_ranks[sample, indices],
            n_bins=n_bins,
        )

    alpha = (1.0 - ci_prob) / 2.0
    finite_pe = np.isfinite(pe_draws)
    denom = finite_pe.sum(axis=0)
    numer = ((pe_draws > 1) & finite_pe).sum(axis=0)
    p_gt_one = np.divide(
        numer,
        denom,
        out=np.full(n_bins, np.nan, dtype=float),
        where=denom > 0,
    )

    curve_summary = pd.DataFrame(
        {
            "rank": rank,
            "pe_mean": np.nanmean(pe_draws, axis=0),
            "pe_median": np.nanmedian(pe_draws, axis=0),
            "pe_lower": np.nanquantile(pe_draws, alpha, axis=0),
            "pe_upper": np.nanquantile(pe_draws, 1.0 - alpha, axis=0),
            "pe_q25": np.nanquantile(pe_draws, 0.25, axis=0),
            "pe_q75": np.nanquantile(pe_draws, 0.75, axis=0),
            "p_pe_gt_one": p_gt_one,
        }
    )

    finite = boyce_draws[np.isfinite(boyce_draws)]
    if len(finite):
        boyce_summary = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "lower": float(np.quantile(finite, alpha)),
            "upper": float(np.quantile(finite, 1.0 - alpha)),
            "p_gt_zero": float(np.mean(finite > 0)),
            "n_draws": int(len(finite)),
        }
    else:
        boyce_summary = {
            "mean": np.nan,
            "median": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "p_gt_zero": np.nan,
            "n_draws": 0,
        }

    return {
        "curve_summary": curve_summary,
        "pe_draws": pe_draws,
        "boyce_draws": boyce_draws,
        "boyce_summary": boyce_summary,
    }


def _run_bootstrap_validation(
    used_ranks: np.ndarray,
    timestamps,
    *,
    heldout_id,
    outer_fold,
    n_bins: int,
    n_replicates: int,
    block: str | None,
    seed: int,
) -> tuple[list[dict], list[pd.DataFrame], dict]:
    """Block-bootstrap complete temporal blocks with replacement."""
    rng = np.random.default_rng(seed)
    n_posterior = used_ranks.shape[0]

    block_label = fixed_duration_block_labels(timestamps, block)
    block_frame = pd.DataFrame({"_block": block_label})
    block_indices = {
        block_id: group.index.to_numpy()
        for block_id, group in block_frame.groupby("_block", sort=True)
    }
    block_ids = np.array(list(block_indices), dtype=object)
    if len(block_ids) < 2:
        raise ValueError(
            "Bootstrap validation requires at least two temporal blocks."
        )

    summary_rows: list[dict] = []
    curve_rows: list[pd.DataFrame] = []

    for replicate in range(n_replicates):
        posterior_draw = int(rng.integers(n_posterior))

        sampled_blocks = rng.choice(
            block_ids,
            size=len(block_ids),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [block_indices[block_id] for block_id in sampled_blocks]
        )

        rank, pe_ratio, boyce = boyce_curve_from_ranks(
            used_ranks[posterior_draw, sampled_indices],
            n_bins=n_bins,
        )

        sampled_block_labels = [str(value) for value in sampled_blocks]
        summary_rows.append(
            {
                "heldout_ID": heldout_id,
                "outer_fold": outer_fold,
                "method": "bootstrap",
                "replicate": replicate,
                "replicate_label": f"B{replicate:04d}",
                "heldout_units": np.nan,
                "n_used": int(len(sampled_indices)),
                "n_temporal_units": int(len(sampled_blocks)),
                "n_unique_temporal_units": int(
                    len(set(sampled_block_labels))
                ),
                "start": pd.NaT,
                "end": pd.NaT,
                "posterior_draw": posterior_draw,
                "boyce": boyce,
                "boyce_lower": np.nan,
                "boyce_upper": np.nan,
                "p_boyce_gt_zero": np.nan,
            }
        )
        curve_rows.append(
            pd.DataFrame(
                {
                    "heldout_ID": heldout_id,
                    "outer_fold": outer_fold,
                    "method": "bootstrap",
                    "replicate": replicate,
                    "replicate_label": f"B{replicate:04d}",
                    "rank": rank,
                    "pe_ratio": pe_ratio,
                    "pe_lower": np.nan,
                    "pe_upper": np.nan,
                    "p_pe_gt_one": np.nan,
                }
            )
        )

    raw = {
        "bootstrap_block": block,
        "n_source_blocks": int(len(block_ids)),
        "source_blocks": [str(value) for value in block_ids],
    }
    return summary_rows, curve_rows, raw


def _run_contiguous_temporal_validation(
    used_ranks: np.ndarray,
    timestamps,
    *,
    heldout_id,
    outer_fold,
    k_folds: int,
    temporal_unit: str,
    n_bins: int,
    ci_prob: float,
) -> tuple[list[dict], list[pd.DataFrame], dict]:
    """Evaluate consecutive real temporal periods and record exact units."""
    folds, temporal = assign_contiguous_temporal_folds(
        timestamps,
        k=k_folds,
        unit=temporal_unit,
    )
    ts = temporal["Timestamp"]
    summary_rows: list[dict] = []
    curve_rows: list[pd.DataFrame] = []
    fold_results: dict[int, Any] = {}

    for temporal_fold in range(k_folds):
        indices = np.flatnonzero(folds == temporal_fold)
        if len(indices) == 0:
            continue

        result = _evaluate_posterior_subset(
            used_ranks,
            indices,
            n_bins=n_bins,
            ci_prob=ci_prob,
        )
        summary = result["boyce_summary"]

        fold_units = (
            temporal.iloc[indices][["temporal_unit", "temporal_label"]]
            .drop_duplicates()
            .sort_values("temporal_unit")
        )
        heldout_labels = fold_units["temporal_label"].astype(str).tolist()
        heldout_units = ", ".join(heldout_labels)
        if not heldout_labels:
            unit_range = "empty"
        elif len(heldout_labels) == 1:
            unit_range = heldout_labels[0]
        else:
            unit_range = f"{heldout_labels[0]} → {heldout_labels[-1]}"
        replicate_label = f"F{temporal_fold:02d} | {unit_range}"

        summary_rows.append(
            {
                "heldout_ID": heldout_id,
                "outer_fold": outer_fold,
                "method": "temporal_contiguous",
                "replicate": temporal_fold,
                "replicate_label": replicate_label,
                "heldout_units": heldout_units,
                "n_used": int(len(indices)),
                "n_temporal_units": int(len(heldout_labels)),
                "n_unique_temporal_units": int(len(heldout_labels)),
                "start": ts.iloc[indices].min(),
                "end": ts.iloc[indices].max(),
                "posterior_draw": np.nan,
                "boyce": summary["median"],
                "boyce_lower": summary["lower"],
                "boyce_upper": summary["upper"],
                "p_boyce_gt_zero": summary["p_gt_zero"],
            }
        )

        curve = result["curve_summary"].copy()
        curve["heldout_ID"] = heldout_id
        curve["outer_fold"] = outer_fold
        curve["method"] = "temporal_contiguous"
        curve["replicate"] = temporal_fold
        curve["replicate_label"] = replicate_label
        curve["heldout_units"] = heldout_units
        curve["pe_ratio"] = curve["pe_median"]
        curve_rows.append(
            curve[
                [
                    "heldout_ID",
                    "outer_fold",
                    "method",
                    "replicate",
                    "replicate_label",
                    "heldout_units",
                    "rank",
                    "pe_ratio",
                    "pe_lower",
                    "pe_upper",
                    "p_pe_gt_one",
                ]
            ]
        )
        fold_results[temporal_fold] = result

    assignments = temporal.copy()
    assignments["temporal_fold"] = folds
    return (
        summary_rows,
        curve_rows,
        {
            "assignments": assignments,
            "fold_results": fold_results,
        },
    )


def _summarize_replicates(
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    if replicates.empty:
        return pd.DataFrame()

    rows = []
    for (heldout_id, method), data in replicates.groupby(
        ["heldout_ID", "method"],
        sort=False,
    ):
        values = data["boyce"].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        rows.append(
            {
                "heldout_ID": heldout_id,
                "method": method,
                "n_replicates": int(len(values)),
                "boyce_mean": float(np.mean(values)),
                "boyce_median": float(np.median(values)),
                "boyce_q025": float(np.quantile(values, 0.025)),
                "boyce_q10": float(np.quantile(values, 0.10)),
                "boyce_q25": float(np.quantile(values, 0.25)),
                "boyce_q75": float(np.quantile(values, 0.75)),
                "boyce_q90": float(np.quantile(values, 0.90)),
                "boyce_q975": float(np.quantile(values, 0.975)),
                "p_boyce_gt_zero": float(np.mean(values > 0)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_curves(
    curves: pd.DataFrame,
) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()

    rows = []
    for (heldout_id, method, rank), data in curves.groupby(
        ["heldout_ID", "method", "rank"],
        sort=False,
    ):
        values = data["pe_ratio"].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        rows.append(
            {
                "heldout_ID": heldout_id,
                "method": method,
                "rank": float(rank),
                "n_replicates": int(len(values)),
                "pe_mean": float(np.mean(values)),
                "pe_median": float(np.median(values)),
                "pe_q025": float(np.quantile(values, 0.025)),
                "pe_q10": float(np.quantile(values, 0.10)),
                "pe_q25": float(np.quantile(values, 0.25)),
                "pe_q75": float(np.quantile(values, 0.75)),
                "pe_q90": float(np.quantile(values, 0.90)),
                "pe_q975": float(np.quantile(values, 0.975)),
                "p_pe_gt_one": float(np.mean(values > 1)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_bayesian_loio_uncertainty(
    diagnostics: Mapping,
    *,
    methods: Sequence[str] = ("bootstrap", "temporal_contiguous"),
    n_bins: int = 20,
    bootstrap_replicates: int = 2000,
    bootstrap_block: str | None = "7D",
    temporal_folds: int = 10,
    temporal_unit: str = "W",
    ci_prob: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Post-fit uncertainty/temporal-stability evaluation for Bayesian LOIO."""
    allowed = {"bootstrap", "temporal_contiguous"}
    methods = tuple(methods)
    unknown = set(methods).difference(allowed)
    if unknown:
        if "temporal_random" in unknown:
            raise ValueError(
                "'temporal_random' was removed as redundant with the "
                "temporally blocked bootstrap; use 'bootstrap' and/or "
                "'temporal_contiguous'."
            )
        raise ValueError(f"Unknown validation methods: {sorted(unknown)}")
    if n_bins < 3:
        raise ValueError("n_bins must be at least 3.")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    if temporal_folds < 2:
        raise ValueError("temporal_folds must be at least 2.")

    baseline_rows: list[dict] = []
    baseline_curve_rows: list[pd.DataFrame] = []
    replicate_rows: list[dict] = []
    curve_rows: list[pd.DataFrame] = []
    raw: dict[Any, Any] = {}

    for index, (heldout_id, diagnostic) in enumerate(diagnostics.items()):
        if "scores" not in diagnostic:
            raise KeyError(
                f"diagnostics[{heldout_id!r}] does not contain 'scores'."
            )

        scores = diagnostic["scores"]
        required = {"used_scores", "available_scores", "used_data"}
        missing = required.difference(scores)
        if missing:
            raise KeyError(
                f"diagnostics[{heldout_id!r}]['scores'] is missing: "
                f"{sorted(missing)}"
            )

        used_scores = np.asarray(scores["used_scores"], dtype=float)
        available_scores = np.asarray(
            scores["available_scores"],
            dtype=float,
        )
        used_data = scores["used_data"].reset_index(drop=True).copy()

        if len(used_data) != used_scores.shape[1]:
            raise ValueError(
                f"{heldout_id}: used_data has {len(used_data)} rows but "
                f"used_scores has {used_scores.shape[1]} locations."
            )
        if "Timestamp" not in used_data.columns:
            raise KeyError(
                f"{heldout_id}: used_data does not contain 'Timestamp'."
            )

        outer_fold = diagnostic.get("fold", np.nan)
        used_ranks = _used_percentile_ranks(
            used_scores,
            available_scores,
        )
        baseline = _evaluate_posterior_subset(
            used_ranks,
            np.arange(used_ranks.shape[1]),
            n_bins=n_bins,
            ci_prob=ci_prob,
        )
        summary = baseline["boyce_summary"]
        baseline_rows.append(
            {
                "heldout_ID": heldout_id,
                "outer_fold": outer_fold,
                "n_used": int(used_ranks.shape[1]),
                "boyce_mean": summary["mean"],
                "boyce_median": summary["median"],
                "boyce_lower": summary["lower"],
                "boyce_upper": summary["upper"],
                "p_boyce_gt_zero": summary["p_gt_zero"],
            }
        )

        base_curve = baseline["curve_summary"].copy()
        base_curve["heldout_ID"] = heldout_id
        base_curve["outer_fold"] = outer_fold
        baseline_curve_rows.append(base_curve)
        raw[heldout_id] = {"baseline": baseline}

        individual_seed = seed + 100_000 * index

        if "bootstrap" in methods:
            rows, curves_i, raw_i = _run_bootstrap_validation(
                used_ranks,
                used_data["Timestamp"],
                heldout_id=heldout_id,
                outer_fold=outer_fold,
                n_bins=n_bins,
                n_replicates=bootstrap_replicates,
                block=bootstrap_block,
                seed=individual_seed + 1_000,
            )
            replicate_rows.extend(rows)
            curve_rows.extend(curves_i)
            raw[heldout_id]["bootstrap"] = raw_i

        if "temporal_contiguous" in methods:
            rows, curves_i, raw_i = _run_contiguous_temporal_validation(
                used_ranks,
                used_data["Timestamp"],
                heldout_id=heldout_id,
                outer_fold=outer_fold,
                k_folds=temporal_folds,
                temporal_unit=temporal_unit,
                n_bins=n_bins,
                ci_prob=ci_prob,
            )
            replicate_rows.extend(rows)
            curve_rows.extend(curves_i)
            raw[heldout_id]["temporal_contiguous"] = raw_i

    baseline_summary = pd.DataFrame(baseline_rows)
    baseline_curves = (
        pd.concat(baseline_curve_rows, ignore_index=True)
        if baseline_curve_rows
        else pd.DataFrame()
    )
    replicate_summary = pd.DataFrame(replicate_rows)
    curves = (
        pd.concat(curve_rows, ignore_index=True)
        if curve_rows
        else pd.DataFrame()
    )

    return {
        "baseline_summary": baseline_summary,
        "baseline_curves": baseline_curves,
        "replicate_summary": replicate_summary,
        "method_summary": _summarize_replicates(replicate_summary),
        "curves": curves,
        "curve_summary": _summarize_curves(curves),
        "raw": raw,
        "config": {
            "methods": methods,
            "n_bins": n_bins,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_block": bootstrap_block,
            "temporal_folds": temporal_folds,
            "temporal_unit": temporal_unit,
            "ci_prob": ci_prob,
            "seed": seed,
        },
    }


def plot_bayesian_loio_uncertainty(
    validation: Mapping,
    heldout_id,
    *,
    figsize: tuple[float, float] = (12, 5),
):
    """Compare blocked-bootstrap uncertainty with contiguous temporal curves."""
    import matplotlib.pyplot as plt

    curves = validation["curves"]
    summary = validation["curve_summary"]
    replicates = validation["replicate_summary"]

    fig, (ax_boot, ax_time) = plt.subplots(1, 2, figsize=figsize)

    boot = summary.loc[
        (summary["heldout_ID"] == heldout_id)
        & (summary["method"] == "bootstrap")
    ].sort_values("rank")
    if not boot.empty:
        ax_boot.fill_between(
            boot["rank"],
            boot["pe_q025"],
            boot["pe_q975"],
            alpha=0.12,
            label="95% bootstrap interval",
        )
        ax_boot.fill_between(
            boot["rank"],
            boot["pe_q25"],
            boot["pe_q75"],
            alpha=0.20,
            label="Bootstrap IQR",
        )
        ax_boot.plot(
            boot["rank"],
            boot["pe_median"],
            linewidth=2.2,
            label="Bootstrap median",
        )

    ax_boot.axhline(1, linestyle="--", linewidth=1)
    ax_boot.set_title("Blocked-bootstrap validation uncertainty")
    ax_boot.set_xlabel("Predicted selection rank")
    ax_boot.set_ylabel("P/E")
    ax_boot.legend(frameon=False)

    temporal = curves.loc[
        (curves["heldout_ID"] == heldout_id)
        & (curves["method"] == "temporal_contiguous")
    ]
    for label, data in temporal.groupby("replicate_label", sort=False):
        data = data.sort_values("rank")
        ax_time.plot(
            data["rank"],
            data["pe_ratio"],
            linewidth=1.2,
            alpha=0.65,
            label=label,
        )

    ax_time.axhline(1, linestyle="--", linewidth=1)
    ax_time.set_title("Contiguous temporal validation")
    ax_time.set_xlabel("Predicted selection rank")
    ax_time.set_ylabel("P/E")
    if temporal["replicate_label"].nunique() <= 10:
        ax_time.legend(frameon=False, fontsize=7)

    temporal_periods = replicates.loc[
        (replicates["heldout_ID"] == heldout_id)
        & (replicates["method"] == "temporal_contiguous"),
        ["replicate", "replicate_label", "heldout_units", "boyce"],
    ].copy()

    fig.suptitle(str(heldout_id))
    fig.tight_layout()
    return fig, (ax_boot, ax_time), temporal_periods


__all__ = [
    "prepare_bayesian_boyce_scores",
    "bayesian_boyce_quantile_scores",
    "plot_bayesian_boyce",
    "evaluate_bayesian_loio_uncertainty",
    "plot_bayesian_loio_uncertainty",
]
