"""Post-fit validation uncertainty for frequentist leave-one-individual-out RSFs.

The fitted fold-specific RSF is held fixed. A temporally blocked bootstrap
quantifies finite validation-sample uncertainty, while contiguous temporal
blocks diagnose empirical temporal non-stationarity. Availability is sampled
once per held-out individual and reused across all replicates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from hsa._time import require_timezone_aware

from hsa.rsf.validation_utils import (
    assign_contiguous_temporal_folds,
    boyce_curve_from_ranks,
    fixed_duration_block_labels,
)
from hsa.sampling import sample_available_points, sample_raster_stack


def _sample_frame(*args, **kwargs) -> pd.DataFrame:
    """Return the dataframe across old/new sample_raster_stack return contracts."""
    sampled = sample_raster_stack(*args, **kwargs)
    return sampled[0] if isinstance(sampled, tuple) else sampled


def _used_percentile_ranks(
    used_scores: np.ndarray,
    available_scores: np.ndarray,
) -> np.ndarray:
    """Convert used scores to percentile ranks in a fixed availability sample."""
    used = np.asarray(used_scores, dtype=float)
    available = np.asarray(available_scores, dtype=float)
    available = np.sort(available[np.isfinite(available)])
    ranks = np.full(used.shape, np.nan, dtype=np.float32)
    if available.size == 0:
        return ranks
    valid = np.isfinite(used)
    ranks[valid] = (
        np.searchsorted(available, used[valid], side="right") / available.size
    )
    return ranks


def _prepare_frequentist_scores(
    diagnostic: Mapping,
    *,
    n_background: int,
    seed: int,
    pred_col: str = "rsf_pred",
) -> dict[str, Any]:
    """Prepare held-out ranks against one fixed availability sample."""
    required = {"test_pred", "rsf", "domain"}
    missing = required.difference(diagnostic)
    if missing:
        raise KeyError(
            f"Frequentist LOIO diagnostic is missing: {sorted(missing)}"
        )

    used_data = diagnostic["test_pred"].reset_index(drop=True).copy()
    for column in ("Timestamp", pred_col):
        if column not in used_data.columns:
            raise KeyError(
                f"Frequentist LOIO test predictions do not contain {column!r}."
            )

    used_scores = used_data[pred_col].to_numpy(dtype=float)
    valid = np.isfinite(used_scores)
    used_data = used_data.loc[valid].reset_index(drop=True)
    used_scores = used_scores[valid]
    if used_scores.size == 0:
        raise ValueError("No finite held-out RSF predictions are available.")

    timestamps = require_timezone_aware(
        used_data["Timestamp"],
        name="Timestamp",
    )
    invalid_timestamps = timestamps.isna()
    if invalid_timestamps.any():
        raise ValueError(
            "Frequentist LOIO test_pred['Timestamp'] contains "
            f"{int(invalid_timestamps.sum())} missing or invalid value(s) "
            "among locations with finite RSF predictions."
        )
    used_data["Timestamp"] = timestamps

    background_points = sample_available_points(
        diagnostic["domain"],
        n_background,
        seed=seed,
    )
    background_data = _sample_frame(background_points, diagnostic["rsf"])
    if "rsf" not in background_data.columns:
        raise KeyError(
            "Sampling the fitted RSF surface did not produce an 'rsf' column."
        )

    available_scores = background_data["rsf"].to_numpy(dtype=float)
    available_scores = available_scores[np.isfinite(available_scores)]
    if available_scores.size == 0:
        raise ValueError("No finite availability RSF predictions are available.")

    return {
        "used_ranks": _used_percentile_ranks(used_scores, available_scores),
        "used_data": used_data,
        "available_scores": available_scores,
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
    """Block-bootstrap complete temporal blocks with the fitted RSF fixed."""
    rng = np.random.default_rng(seed)
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
        sampled_blocks = rng.choice(
            block_ids,
            size=len(block_ids),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [block_indices[b] for b in sampled_blocks]
        )
        rank, pe_ratio, boyce = boyce_curve_from_ranks(
            used_ranks[sampled_indices],
            n_bins=n_bins,
        )
        labels = [str(b) for b in sampled_blocks]
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
                "n_unique_temporal_units": int(len(set(labels))),
                "start": pd.NaT,
                "end": pd.NaT,
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
                    "heldout_units": np.nan,
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
        "source_blocks": [str(x) for x in block_ids],
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
) -> tuple[list[dict], list[pd.DataFrame], dict]:
    """Evaluate consecutive real temporal periods with the fitted RSF fixed."""
    folds, temporal = assign_contiguous_temporal_folds(
        timestamps,
        k=k_folds,
        unit=temporal_unit,
    )
    ts = temporal["Timestamp"]
    summary_rows: list[dict] = []
    curve_rows: list[pd.DataFrame] = []

    for temporal_fold in range(k_folds):
        indices = np.flatnonzero(folds == temporal_fold)
        if len(indices) == 0:
            continue

        rank, pe_ratio, boyce = boyce_curve_from_ranks(
            used_ranks[indices],
            n_bins=n_bins,
        )
        fold_units = (
            temporal.iloc[indices][["temporal_unit", "temporal_label"]]
            .drop_duplicates()
            .sort_values("temporal_unit")
        )
        labels = fold_units["temporal_label"].astype(str).tolist()
        heldout_units = ", ".join(labels)
        if not labels:
            unit_range = "empty"
        elif len(labels) == 1:
            unit_range = labels[0]
        else:
            unit_range = f"{labels[0]} → {labels[-1]}"
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
                "n_temporal_units": int(len(labels)),
                "n_unique_temporal_units": int(len(labels)),
                "start": ts.iloc[indices].min(),
                "end": ts.iloc[indices].max(),
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
                    "method": "temporal_contiguous",
                    "replicate": temporal_fold,
                    "replicate_label": replicate_label,
                    "heldout_units": heldout_units,
                    "rank": rank,
                    "pe_ratio": pe_ratio,
                    "pe_lower": np.nan,
                    "pe_upper": np.nan,
                    "p_pe_gt_one": np.nan,
                }
            )
        )

    assignments = temporal.copy()
    assignments["temporal_fold"] = folds
    return summary_rows, curve_rows, {"assignments": assignments}


def _summarize_replicates(
    replicates: pd.DataFrame,
    *,
    ci_prob: float,
) -> pd.DataFrame:
    if replicates.empty:
        return pd.DataFrame()
    alpha = (1.0 - ci_prob) / 2.0
    rows = []
    for (heldout_id, method), d in replicates.groupby(
        ["heldout_ID", "method"],
        sort=False,
    ):
        values = d["boyce"].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        rows.append(
            {
                "heldout_ID": heldout_id,
                "method": method,
                "n_replicates": int(len(values)),
                "boyce_mean": float(np.mean(values)),
                "boyce_median": float(np.median(values)),
                "boyce_lower": float(np.quantile(values, alpha)),
                "boyce_upper": float(np.quantile(values, 1.0 - alpha)),
                "boyce_q25": float(np.quantile(values, 0.25)),
                "boyce_q75": float(np.quantile(values, 0.75)),
                "p_boyce_gt_zero": float(np.mean(values > 0)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_curves(
    curves: pd.DataFrame,
    *,
    ci_prob: float,
) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    alpha = (1.0 - ci_prob) / 2.0
    rows = []
    for (heldout_id, method, rank), d in curves.groupby(
        ["heldout_ID", "method", "rank"],
        sort=False,
    ):
        values = d["pe_ratio"].dropna().to_numpy(dtype=float)
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
                "pe_lower": float(np.quantile(values, alpha)),
                "pe_upper": float(np.quantile(values, 1.0 - alpha)),
                "pe_q25": float(np.quantile(values, 0.25)),
                "pe_q75": float(np.quantile(values, 0.75)),
                "p_pe_gt_one": float(np.mean(values > 1.0)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_frequentist_loio_uncertainty(
    diagnostics: Mapping,
    *,
    methods: Sequence[str] = ("bootstrap", "temporal_contiguous"),
    n_bins: int = 20,
    n_background: int = 100_000,
    bootstrap_replicates: int = 2000,
    bootstrap_block: str | None = "7D",
    temporal_folds: int = 10,
    temporal_unit: str = "W",
    ci_prob: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate post-fit validation uncertainty for frequentist LOIO."""
    allowed = {"bootstrap", "temporal_contiguous"}
    methods = tuple(methods)
    unknown = set(methods).difference(allowed)
    if unknown:
        if "temporal_random" in unknown:
            raise ValueError(
                "'temporal_random' is intentionally unsupported; use "
                "'bootstrap' and/or 'temporal_contiguous'."
            )
        raise ValueError(f"Unknown validation methods: {sorted(unknown)}")
    if n_bins < 3:
        raise ValueError("n_bins must be at least 3.")
    if n_background <= 0:
        raise ValueError("n_background must be positive.")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    if temporal_folds < 2:
        raise ValueError("temporal_folds must be at least 2.")
    if not 0 < ci_prob < 1:
        raise ValueError("ci_prob must be between 0 and 1.")

    baseline_rows: list[dict] = []
    baseline_curve_rows: list[pd.DataFrame] = []
    replicate_rows: list[dict] = []
    curve_rows: list[pd.DataFrame] = []
    raw: dict[Any, Any] = {}

    for i, (heldout_id, diagnostic) in enumerate(diagnostics.items()):
        individual_seed = seed + 100_000 * i
        scores = _prepare_frequentist_scores(
            diagnostic,
            n_background=n_background,
            seed=individual_seed,
        )
        used_ranks = scores["used_ranks"]
        used_data = scores["used_data"]
        outer_fold = diagnostic.get("fold", np.nan)

        rank, pe_ratio, boyce = boyce_curve_from_ranks(
            used_ranks,
            n_bins=n_bins,
        )
        baseline_rows.append(
            {
                "heldout_ID": heldout_id,
                "outer_fold": outer_fold,
                "n_used": int(len(used_ranks)),
                "boyce": boyce,
            }
        )
        baseline_curve_rows.append(
            pd.DataFrame(
                {
                    "heldout_ID": heldout_id,
                    "outer_fold": outer_fold,
                    "rank": rank,
                    "pe_ratio": pe_ratio,
                }
            )
        )
        raw[heldout_id] = {
            "baseline": {
                "used_ranks": used_ranks,
                "rank": rank,
                "pe_ratio": pe_ratio,
                "boyce": boyce,
                "available_scores": scores["available_scores"],
            }
        }

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
        "method_summary": _summarize_replicates(
            replicate_summary,
            ci_prob=ci_prob,
        ),
        "curves": curves,
        "curve_summary": _summarize_curves(
            curves,
            ci_prob=ci_prob,
        ),
        "raw": raw,
        "config": {
            "methods": methods,
            "n_bins": n_bins,
            "n_background": n_background,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_block": bootstrap_block,
            "temporal_folds": temporal_folds,
            "temporal_unit": temporal_unit,
            "ci_prob": ci_prob,
            "seed": seed,
            "uncertainty_scope": "validation_sample_only",
        },
    }


def plot_frequentist_loio_uncertainty(
    validation: Mapping,
    heldout_id,
    *,
    figsize: tuple[float, float] = (12, 5),
):
    """Plot bootstrap uncertainty and contiguous temporal P/E curves."""
    curves = validation["curves"]
    summary = validation["curve_summary"]
    replicates = validation["replicate_summary"]
    ci_prob = validation.get("config", {}).get("ci_prob", 0.95)

    required_summary_columns = {
        "heldout_ID",
        "method",
        "rank",
        "pe_median",
        "pe_lower",
        "pe_upper",
        "pe_q25",
        "pe_q75",
    }
    if summary.empty:
        raise ValueError(
            "Cannot plot frequentist LOIO uncertainty because curve_summary "
            "is empty. Run at least one uncertainty strategy on non-empty "
            "LOIO diagnostics first."
        )
    missing_summary_columns = required_summary_columns.difference(summary.columns)
    if missing_summary_columns:
        raise ValueError(
            "Cannot plot frequentist LOIO uncertainty because curve_summary "
            "is missing required columns: "
            f"{sorted(missing_summary_columns)}"
        )

    if curves.empty:
        raise ValueError(
            "Cannot plot frequentist LOIO uncertainty because curves is empty."
        )

    required_curve_columns = {
        "heldout_ID",
        "method",
        "replicate_label",
        "rank",
        "pe_ratio",
    }
    missing_curve_columns = required_curve_columns.difference(curves.columns)
    if missing_curve_columns:
        raise ValueError(
            "Cannot plot frequentist LOIO uncertainty because curves is "
            f"missing required columns: {sorted(missing_curve_columns)}"
        )

    if replicates.empty:
        raise ValueError(
            "Cannot plot frequentist LOIO uncertainty because "
            "replicate_summary is empty."
        )

    required_replicate_columns = {
        "heldout_ID",
        "method",
        "replicate",
        "replicate_label",
        "heldout_units",
        "boyce",
    }
    missing_replicate_columns = required_replicate_columns.difference(
        replicates.columns
    )
    if missing_replicate_columns:
        raise ValueError(
            "Cannot plot frequentist LOIO uncertainty because "
            "replicate_summary is missing required columns: "
            f"{sorted(missing_replicate_columns)}"
        )

    available_ids = set(summary["heldout_ID"]).union(curves["heldout_ID"])
    if heldout_id not in available_ids:
        raise KeyError(
            f"heldout_id={heldout_id!r} is not present in validation output."
        )

    import matplotlib.pyplot as plt

    fig, (ax_boot, ax_time) = plt.subplots(1, 2, figsize=figsize)

    boot = summary.loc[
        (summary["heldout_ID"] == heldout_id)
        & (summary["method"] == "bootstrap")
    ].sort_values("rank")
    if not boot.empty:
        ax_boot.fill_between(
            boot["rank"],
            boot["pe_lower"],
            boot["pe_upper"],
            alpha=0.12,
            label=f"{ci_prob:.0%} bootstrap interval",
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
    if not boot.empty:
        ax_boot.legend(frameon=False)

    temporal = curves.loc[
        (curves["heldout_ID"] == heldout_id)
        & (curves["method"] == "temporal_contiguous")
    ]
    for label, d in temporal.groupby("replicate_label", sort=False):
        d = d.sort_values("rank")
        ax_time.plot(
            d["rank"],
            d["pe_ratio"],
            linewidth=1.2,
            alpha=0.65,
            label=label,
        )
    ax_time.axhline(1, linestyle="--", linewidth=1)
    ax_time.set_title("Contiguous temporal validation")
    ax_time.set_xlabel("Predicted selection rank")
    ax_time.set_ylabel("P/E")
    if not temporal.empty and temporal["replicate_label"].nunique() <= 10:
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
    "evaluate_frequentist_loio_uncertainty",
    "plot_frequentist_loio_uncertainty",
]
