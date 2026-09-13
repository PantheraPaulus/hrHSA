from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def common_language_effect_size(vals_used, vals_available) -> float:
    """Estimate P(used value > available value) with a Mann-Whitney U statistic.

    Values near 0.5 indicate little directional separation. Values above 0.5
    indicate used values tend to be larger than available values; values below
    0.5 indicate the reverse.
    """

    from scipy.stats import mannwhitneyu

    vals_used = np.asarray(vals_used)
    vals_available = np.asarray(vals_available)

    vals_used = vals_used[np.isfinite(vals_used)]
    vals_available = vals_available[np.isfinite(vals_available)]

    if len(vals_used) == 0 or len(vals_available) == 0:
        return float(np.nan)

    u_stat, _ = mannwhitneyu(vals_used, vals_available, alternative="two-sided")
    return float(u_stat / (len(vals_used) * len(vals_available)))


def summarize_variable_use_available(
    sampled: pd.DataFrame,
    variable: str,
    *,
    used_col: str = "used",
    id_col: str | None = None,
    threshold: float | None = None,
    threshold_direction: str = ">",
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Summarize one predictor for used and available samples.

    The returned table contains sample sizes, proportions above/below a chosen
    threshold, distribution summaries, the Mann-Whitney U statistic, its p-value,
    and the common-language effect size (CLES), i.e. P(used > available).

    If ``id_col`` is supplied, the summary is calculated separately for each
    individual. Otherwise, a single pooled row is returned.
    """

    from scipy.stats import mannwhitneyu

    required = {variable, used_col}
    if id_col is not None:
        required.add(id_col)
    missing = required.difference(sampled.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if threshold_direction not in {">", ">=", "<", "<="}:
        raise ValueError("threshold_direction must be one of '>', '>=', '<', '<='.")

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be one of 'two-sided', 'less', or 'greater'.")

    def _clean(values) -> np.ndarray:
        values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return values[np.isfinite(values)]

    def _prop(values: np.ndarray) -> float:
        if threshold is None or values.size == 0:
            return float(np.nan)
        if threshold_direction == ">":
            return float(np.mean(values > threshold))
        if threshold_direction == ">=":
            return float(np.mean(values >= threshold))
        if threshold_direction == "<":
            return float(np.mean(values < threshold))
        return float(np.mean(values <= threshold))

    def _one_group(group: pd.DataFrame, group_id=None) -> dict[str, Any]:
        vals_used = _clean(group.loc[group[used_col] == True, variable])
        vals_available = _clean(group.loc[group[used_col] != True, variable])

        row: dict[str, Any] = {
            "variable": variable,
            "n_used": int(vals_used.size),
            "n_available": int(vals_available.size),
            "mean_used": float(np.mean(vals_used)) if vals_used.size else np.nan,
            "mean_available": float(np.mean(vals_available)) if vals_available.size else np.nan,
            "median_used": float(np.median(vals_used)) if vals_used.size else np.nan,
            "median_available": float(np.median(vals_available)) if vals_available.size else np.nan,
            "prop_used": _prop(vals_used),
            "prop_available": _prop(vals_available),
            "threshold": threshold,
            "threshold_direction": threshold_direction if threshold is not None else None,
            "mannwhitney_u": np.nan,
            "mannwhitney_pvalue": np.nan,
            "cles": np.nan,
        }

        if group_id is not None and id_col is not None:
            row[id_col] = group_id

        if vals_used.size > 0 and vals_available.size > 0:
            test = mannwhitneyu(vals_used, vals_available, alternative=alternative)
            row["mannwhitney_u"] = float(test.statistic)
            row["mannwhitney_pvalue"] = float(test.pvalue)
            row["cles"] = float(test.statistic / (vals_used.size * vals_available.size))

        row["delta_mean"] = row["mean_used"] - row["mean_available"]
        row["delta_median"] = row["median_used"] - row["median_available"]
        row["delta_prop"] = row["prop_used"] - row["prop_available"]
        row["cles_centered"] = abs(row["cles"] - 0.5) if np.isfinite(row["cles"]) else np.nan
        return row

    if id_col is None:
        return pd.DataFrame([_one_group(sampled)])

    rows = [_one_group(group, group_id) for group_id, group in sampled.groupby(id_col)]
    out = pd.DataFrame(rows)
    cols = [id_col] + [col for col in out.columns if col != id_col]
    return out[cols]


def summarize_continuous(vals_used, vals_available) -> dict[str, Any]:
    """Summarize used-vs-available separation for one continuous predictor."""

    from scipy.stats import ks_2samp, wasserstein_distance

    vals_used = np.asarray(vals_used)
    vals_available = np.asarray(vals_available)

    vals_used = vals_used[np.isfinite(vals_used)]
    vals_available = vals_available[np.isfinite(vals_available)]

    if len(vals_used) == 0 or len(vals_available) == 0:
        return {
            "type": "continuous",
            "n_used": len(vals_used),
            "n_available": len(vals_available),
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "wasserstein": np.nan,
            "cles": np.nan,
            "mean_used": np.nan,
            "mean_available": np.nan,
            "median_used": np.nan,
            "median_available": np.nan,
            "delta_mean": np.nan,
            "delta_median": np.nan,
        }

    ks = ks_2samp(vals_used, vals_available)
    wd = wasserstein_distance(vals_used, vals_available)
    cles = common_language_effect_size(vals_used, vals_available)

    mean_used = float(np.mean(vals_used))
    mean_available = float(np.mean(vals_available))
    median_used = float(np.median(vals_used))
    median_available = float(np.median(vals_available))

    return {
        "type": "continuous",
        "n_used": len(vals_used),
        "n_available": len(vals_available),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(wd),
        "cles": float(cles),
        "mean_used": mean_used,
        "mean_available": mean_available,
        "median_used": median_used,
        "median_available": median_available,
        "delta_mean": mean_used - mean_available,
        "delta_median": median_used - median_available,
    }


def summarize_categorical(vals_used, vals_available) -> dict[str, Any]:
    """Summarize used-vs-available separation for one categorical predictor."""

    used = pd.Series(vals_used).dropna()
    available = pd.Series(vals_available).dropna()
    cats = sorted(set(used.unique()).union(set(available.unique())))

    if len(cats) == 0:
        return {
            "type": "categorical",
            "n_used": len(used),
            "n_available": len(available),
            "tv_distance": np.nan,
            "n_categories": 0,
        }

    p_used = used.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    p_available = available.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    tv_distance = 0.5 * np.abs(p_used - p_available).sum()

    return {
        "type": "categorical",
        "n_used": len(used),
        "n_available": len(available),
        "tv_distance": float(tv_distance),
        "n_categories": len(cats),
    }


def plot_continuous_ecdfs(
    variable: str,
    subset: pd.DataFrame,
    stats_row: pd.Series | dict[str, Any],
    *,
    used_col: str = "used",
    value_col: str = "value",
    figsize: tuple[float, float] = (12, 4),
):
    """Plot used and available ECDFs for one continuous predictor."""

    import matplotlib.pyplot as plt

    vals_used = subset.loc[subset[used_col] == True, value_col].to_numpy()
    vals_available = subset.loc[subset[used_col] != True, value_col].to_numpy()

    vals_used = vals_used[np.isfinite(vals_used)]
    vals_available = vals_available[np.isfinite(vals_available)]

    fig, axes = plt.subplots(figsize=figsize, nrows=2, sharex=True, sharey=True)
    fig.suptitle(variable)

    if len(vals_used) > 0:
        used_sorted = np.sort(vals_used)
        used_y = np.arange(1, len(used_sorted) + 1) / len(used_sorted)
        axes[0].step(used_sorted, used_y, where="pre")
        axes[0].fill_between(used_sorted, used_y, step="pre", alpha=0.3)
    axes[0].set_title("Used")
    axes[0].set_ylabel("ECDF")

    if len(vals_available) > 0:
        available_sorted = np.sort(vals_available)
        available_y = np.arange(1, len(available_sorted) + 1) / len(available_sorted)
        axes[1].step(available_sorted, available_y, where="pre")
        axes[1].fill_between(available_sorted, available_y, step="pre", alpha=0.3)
    axes[1].set_title("Available")
    axes[1].set_ylabel("ECDF")
    axes[1].set_xlabel("Value")

    annotation = (
        f"KS={stats_row['ks_stat']:.3f} | "
        f"W={stats_row['wasserstein']:.3f} | "
        f"Δmed={stats_row['delta_median']:.3f} | "
        f"CLES={stats_row['cles']:.3f}"
    )
    fig.text(0.99, 0.01, annotation, ha="right", va="bottom", fontsize=10)
    fig.tight_layout()
    return fig, axes


def plot_categorical_bars(
    variable: str,
    subset: pd.DataFrame,
    stats_row: pd.Series | dict[str, Any],
    *,
    used_col: str = "used",
    value_col: str = "value",
    normalize: bool = True,
    figsize: tuple[float, float] = (12, 4),
):
    """Plot used and available category frequencies for one categorical predictor."""

    import matplotlib.pyplot as plt

    vals_used = pd.Series(subset.loc[subset[used_col] == True, value_col]).dropna()
    vals_available = pd.Series(subset.loc[subset[used_col] != True, value_col]).dropna()
    cats = sorted(set(vals_used.unique()).union(set(vals_available.unique())))

    used_counts = vals_used.value_counts(normalize=normalize).reindex(cats, fill_value=0.0)
    available_counts = vals_available.value_counts(normalize=normalize).reindex(cats, fill_value=0.0)

    plot_df = pd.DataFrame(
        {
            "category": cats,
            "Used": used_counts.values,
            "Available": available_counts.values,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(plot_df))
    width = 0.4

    ax.bar(x - width / 2, plot_df["Used"], width, label="Used", alpha=0.8)
    ax.bar(x + width / 2, plot_df["Available"], width, label="Available", alpha=0.8)

    ax.set_title(variable)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["category"], rotation=45, ha="right")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.legend()

    annotation = f"TV distance={stats_row['tv_distance']:.3f} | k={stats_row['n_categories']}"
    fig.text(0.99, 0.01, annotation, ha="right", va="bottom", fontsize=10)
    fig.tight_layout()
    return fig, ax


def inspect_predictors(
    long: pd.DataFrame,
    categorical_predictors: Iterable[str] | None = None,
    *,
    variable_col: str = "variable",
    value_col: str = "value",
    used_col: str = "used",
    sort_by: str = "ks_stat",
    ascending: bool = False,
    plot: bool = True,
    plot_top_n: int | None = None,
    include_categorical: bool = True,
    include_continuous: bool = True,
) -> pd.DataFrame:
    """Inspect used-vs-available separation in a long predictor table.

    Parameters
    ----------
    long:
        Long table with at least ``variable_col``, ``value_col``, and ``used_col``.
    categorical_predictors:
        Names in ``variable_col`` that should be treated as categorical.
    plot:
        If True, create notebook-friendly diagnostic plots. Set False on HPC or
        in non-interactive workflows.

    Returns
    -------
    pandas.DataFrame
        One row per predictor with divergence metrics and ranking columns.
    """

    required = {variable_col, value_col, used_col}
    missing = required.difference(long.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    categorical_predictors = set(categorical_predictors or [])
    summary_rows = []

    for variable, subset in long.groupby(variable_col):
        vals_used = subset.loc[subset[used_col] == True, value_col].values
        vals_available = subset.loc[subset[used_col] != True, value_col].values

        if variable in categorical_predictors:
            stats = summarize_categorical(vals_used, vals_available)
        else:
            stats = summarize_continuous(vals_used, vals_available)

        stats["variable"] = variable
        summary_rows.append(stats)

    summary = pd.DataFrame(summary_rows)

    if "delta_median" in summary.columns:
        summary["abs_delta_median"] = summary["delta_median"].abs()
    if "delta_mean" in summary.columns:
        summary["abs_delta_mean"] = summary["delta_mean"].abs()
    if "cles" in summary.columns:
        summary["cles_centered"] = (summary["cles"] - 0.5).abs()

    if sort_by not in summary.columns:
        raise ValueError(f"sort_by={sort_by!r} not found in summary columns: {list(summary.columns)}")

    summary = summary.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    if plot:
        plot_order = summary["variable"].tolist()
        if plot_top_n is not None:
            plot_order = plot_order[:plot_top_n]

        for variable in plot_order:
            row = summary.loc[summary["variable"] == variable].iloc[0]
            subset = long.loc[long[variable_col] == variable]

            if row["type"] == "categorical":
                if include_categorical:
                    plot_categorical_bars(variable, subset, row, used_col=used_col, value_col=value_col)
            elif include_continuous:
                plot_continuous_ecdfs(variable, subset, row, used_col=used_col, value_col=value_col)

    return summary

def inspect_sampled_use_available(
    sampled: pd.DataFrame,
    columns: str | Iterable[str] | None = None,
    *,
    used_col: str = "used",
    id_col: str | None = None,
    exclude_columns: Iterable[str] = (
        "x",
        "y",
        "geometry",
        "Timestamp",
        "time_rounded",
    ),
    sort_by: str = "ks_stat",
    ascending: bool = False,
    plot: bool = True,
    plot_top_n: int | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> dict[str, Any]:
    """Inspect continuous used-vs-available predictors from a sampled wide table.

    Parameters
    ----------
    sampled:
        Wide sampled table, e.g. columns such as predictors, x, y, used,
        Timestamp, Individual_ID.
    columns:
        Predictor columns to inspect. If None, all numeric columns are used
        except metadata columns.
    used_col:
        Boolean column where True = used and False = available.
    id_col:
        Optional individual ID column. If supplied, statistics are calculated
        by individual, but plots remain pooled.
    exclude_columns:
        Metadata columns ignored when columns=None.
    sort_by:
        Summary column used for sorting.
    ascending:
        Sort direction.
    plot:
        Whether to create pooled ECDF plots.
    plot_top_n:
        If supplied, only plot the top n predictors after pooled sorting.
    figsize:
        Figure size for each predictor plot.

    Returns
    -------
    dict
        {
            "stats": statistics table,
            "pooled_stats": pooled statistics table,
            "figures": dict of variable -> (fig, axes)
        }
    """

    required = {used_col}
    if id_col is not None:
        required.add(id_col)

    missing = required.difference(sampled.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ------------------------------------------------------------
    # Infer predictor columns
    # ------------------------------------------------------------
    if columns is None:
        excluded = set(exclude_columns)
        excluded.add(used_col)

        if id_col is not None:
            excluded.add(id_col)

        columns = [
            col
            for col in sampled.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(sampled[col])
        ]

    elif isinstance(columns, str):
        columns = [columns]

    else:
        columns = list(columns)

    if not columns:
        raise ValueError("No predictor columns supplied or inferred.")

    missing_columns = [col for col in columns if col not in sampled.columns]
    if missing_columns:
        raise ValueError(f"Predictor columns not found in sampled: {missing_columns}")

    # ------------------------------------------------------------
    # Helper: summarize one variable for one dataframe
    # ------------------------------------------------------------
    def _summarize_one(df: pd.DataFrame, variable: str) -> dict[str, Any]:
        vals_used = df.loc[df[used_col] == True, variable].to_numpy()
        vals_available = df.loc[df[used_col] != True, variable].to_numpy()

        stats = summarize_continuous(vals_used, vals_available)
        stats["variable"] = variable

        return stats

    # ------------------------------------------------------------
    # Pooled statistics
    # ------------------------------------------------------------
    pooled_rows = [
        _summarize_one(sampled, variable)
        for variable in columns
    ]

    pooled_stats = pd.DataFrame(pooled_rows)

    if "delta_median" in pooled_stats.columns:
        pooled_stats["abs_delta_median"] = pooled_stats["delta_median"].abs()
    if "delta_mean" in pooled_stats.columns:
        pooled_stats["abs_delta_mean"] = pooled_stats["delta_mean"].abs()
    if "cles" in pooled_stats.columns:
        pooled_stats["cles_centered"] = (pooled_stats["cles"] - 0.5).abs()

    if sort_by not in pooled_stats.columns:
        raise ValueError(
            f"sort_by={sort_by!r} not found in pooled_stats columns: "
            f"{list(pooled_stats.columns)}"
        )

    pooled_stats = (
        pooled_stats
        .sort_values(sort_by, ascending=ascending)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Individual-level statistics, if requested
    # ------------------------------------------------------------
    if id_col is None:
        stats = pooled_stats.copy()

    else:
        rows = []

        for individual_id, group in sampled.groupby(id_col):
            for variable in columns:
                row = _summarize_one(group, variable)
                row[id_col] = individual_id
                rows.append(row)

        stats = pd.DataFrame(rows)

        if "delta_median" in stats.columns:
            stats["abs_delta_median"] = stats["delta_median"].abs()
        if "delta_mean" in stats.columns:
            stats["abs_delta_mean"] = stats["delta_mean"].abs()
        if "cles" in stats.columns:
            stats["cles_centered"] = (stats["cles"] - 0.5).abs()

        stats = (
            stats
            .sort_values([id_col, sort_by], ascending=[True, ascending])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------
    # Pooled plots
    # ------------------------------------------------------------
    figures = {}

    if plot:
        plot_variables = pooled_stats["variable"].tolist()

        if plot_top_n is not None:
            plot_variables = plot_variables[:plot_top_n]

        for variable in plot_variables:
            pooled_row = pooled_stats.loc[
                pooled_stats["variable"] == variable
            ].iloc[0]

            subset = sampled[[used_col, variable]].rename(
                columns={variable: "value"}
            )

            fig, axes = plot_continuous_ecdfs(
                variable,
                subset,
                pooled_row,
                used_col=used_col,
                value_col="value",
                figsize=figsize,
            )

            figures[variable] = (fig, axes)

    return {
        "stats": stats,
        "pooled_stats": pooled_stats,
        "figures": figures,
    }