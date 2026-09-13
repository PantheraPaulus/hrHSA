from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, beta
import matplotlib.pyplot as plt
from hsa.sampling import sample_available_points, sample_raster_stack


def boyce_quantile_bins(
    pred: pd.DataFrame,
    rsf,
    domain,
    *,
    n_background_points: int = 100_000,
    n_bins: int = 20,
    seed: int = 42,
    pred_col: str = "rsf_pred",
) -> tuple[float, pd.DataFrame]:
    """Continuous Boyce-style index using equal-frequency background bins."""

    bg_points = sample_available_points(domain, n_background_points, seed=seed)
    bg_samples = sample_raster_stack(bg_points, rsf)

    background = np.log(bg_samples["rsf"].to_numpy(dtype=float) + 1e-12)
    used = np.log(pred.loc[pred["used"].astype(bool), pred_col].to_numpy(dtype=float) + 1e-12)
    background = background[np.isfinite(background)]
    used = used[np.isfinite(used)]

    if background.size == 0 or used.size == 0:
        return np.nan, pd.DataFrame(columns=["q_mid", "rsf_mid", "used_n", "bg_n", "pe"])

    edges = np.quantile(background, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.nan, pd.DataFrame(columns=["q_mid", "rsf_mid", "used_n", "bg_n", "pe"])

    used_counts, _ = np.histogram(used, bins=edges)
    bg_counts, _ = np.histogram(background, bins=edges)
    used_prop = used_counts / used.size
    bg_prop = bg_counts / background.size
    pe = np.divide(used_prop, bg_prop, out=np.full_like(used_prop, np.nan, dtype=float), where=bg_prop > 0)

    chart = pd.DataFrame(
        {
            "q_mid": (np.arange(len(pe)) + 0.5) / len(pe),
            "rsf_mid": 0.5 * (edges[:-1] + edges[1:]),
            "used_n": used_counts,
            "bg_n": bg_counts,
            "pe": pe,
        }
    )
    valid = np.isfinite(chart["rsf_mid"]) & np.isfinite(chart["pe"])
    if valid.sum() < 2:
        return np.nan, chart
    boyce, _ = spearmanr(chart.loc[valid, "rsf_mid"], chart.loc[valid, "pe"])
    return float(boyce), chart


def boyce_sliding_window(
    pred: pd.DataFrame,
    rsf,
    domain,
    *,
    n_background_points: int = 100_000,
    window_fraction: float = 0.1,
    step_fraction: float = 0.02,
    seed: int = 42,
    pred_col: str = "rsf_pred",
) -> tuple[float, pd.DataFrame]:
    """Continuous Boyce-style index using overlapping background quantile windows."""

    bg_points = sample_available_points(domain, n_background_points, seed=seed)
    bg_samples = sample_raster_stack(bg_points, rsf)

    background = np.log(bg_samples["rsf"].to_numpy(dtype=float) + 1e-12)
    used = np.log(pred.loc[pred["used"].astype(bool), pred_col].to_numpy(dtype=float) + 1e-12)
    background = background[np.isfinite(background)]
    used = used[np.isfinite(used)]

    rows = []
    for q0 in np.arange(0, 1.0 - window_fraction + 1e-12, step_fraction):
        q1 = q0 + window_fraction
        lo, hi = np.quantile(background, [q0, q1])
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            continue

        in_bg = (background >= lo) & (background < hi)
        in_used = (used >= lo) & (used < hi)
        if in_bg.sum() == 0:
            continue

        pe = (in_used.sum() / used.size) / (in_bg.sum() / background.size)
        rows.append({"rsf_mid": 0.5 * (lo + hi), "pe": pe})

    chart = pd.DataFrame(rows)
    if len(chart) < 2:
        return np.nan, chart
    boyce, _ = spearmanr(chart["rsf_mid"], chart["pe"])
    return float(boyce), chart

def plot_boyce_values(
    summary: pd.DataFrame,
    *,
    id_col: str = "heldout_ID",
    boyce_col: str = "boyce",
    sort: bool = True,
    ax=None,
    figsize: tuple[float, float] = (9, 4),
    title: str = "Boyce validation by held-out individual",
):
    """Plot Boyce index values by fold or held-out individual."""

    df = summary.copy()
    df = df.loc[df[boyce_col].notna()].copy()

    if sort:
        df = df.sort_values(boyce_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(df))

    ax.bar(x, df[boyce_col].to_numpy(dtype=float))
    ax.axhline(0, linewidth=1)
    ax.axhline(df[boyce_col].mean(), linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(df[id_col].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Boyce index")
    ax.set_xlabel(id_col)
    ax.set_title(title)

    ax.text(
        0.99,
        0.02,
        f"mean = {df[boyce_col].mean():.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    fig.tight_layout()
    return fig, ax

def plot_boyce_curves(
    boyce_bins: pd.DataFrame,
    *,
    id_col: str = "heldout_ID",
    x_col: str = "rsf_mid",
    y_col: str = "pe",
    show_points: bool = True,
    show_reference: bool = True,
    log_x: bool = False,
    ax=None,
    figsize: tuple[float, float] = (7, 5),
    title: str = "Boyce calibration curves",
    alpha: float = 0.45,
    legend: bool | str = "auto",
):
    """Plot Boyce calibration curves.

    Expects output from ``boyce_quantile_bins`` or ``boyce_sliding_window``.
    If multiple held-out individuals are present, one curve is drawn per ID.
    """

    df = boyce_bins.copy()

    missing = [col for col in (x_col, y_col) if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])

    if df.empty:
        raise ValueError("No finite Boyce-bin values available for plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if id_col in df.columns:
        groups = df.groupby(id_col)
    else:
        groups = [(None, df)]

    for individual_id, g in groups:
        g = g.sort_values(x_col)

        label = str(individual_id) if individual_id is not None else None

        ax.plot(
            g[x_col].to_numpy(dtype=float),
            g[y_col].to_numpy(dtype=float),
            linewidth=1,
            alpha=alpha,
            label=label,
        )

        if show_points:
            ax.scatter(
                g[x_col].to_numpy(dtype=float),
                g[y_col].to_numpy(dtype=float),
                s=20,
                alpha=alpha,
            )

    if show_reference:
        ax.axhline(1, linestyle="--", linewidth=1)

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel("Log RSF score" if x_col == "rsf_mid" else x_col)
    ax.set_ylabel("Observed / Expected under random use" if y_col == "pe" else y_col)
    ax.set_title(title)

    if legend == "auto":
        legend = id_col in df.columns and df[id_col].nunique() <= 12

    if legend:
        ax.legend(title=id_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    return fig, ax

def calibration_rsf_quantile_bins(
    pred: pd.DataFrame,
    rsf,
    domain,
    *,
    n_background_points: int = 100_000,
    n_bins: int = 10,
    seed: int = 42,
    pred_col: str = "rsf_pred",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Evaluate realized use across equal-area RSF-score quantiles.

    Uniformly sampled background points approximate the available landscape.
    Quantile thresholds are defined from their RSF scores, so each class
    contains approximately the same proportion of available area.

    For each class, the function reports:

    - observed relocation proportion;
    - available-area proportion;
    - probability mass predicted by the fitted static RSF;
    - observed/expected under random use;
    - observed/expected under the fitted static RSF.

    The fitted-RSF comparison assumes uniform baseline accessibility within
    the supplied availability domain.
    """

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    if n_background_points < n_bins:
        raise ValueError(
            "n_background_points must be at least as large as n_bins."
        )

    required = {"used", pred_col}
    missing = required.difference(pred.columns)

    if missing:
        raise KeyError(
            f"Missing required prediction columns: {sorted(missing)}"
        )

    # Uniform points approximate equal-area samples of the domain.
    bg_points = sample_available_points(
        domain,
        n_background_points,
        seed=seed,
    )
    bg_samples = sample_raster_stack(bg_points, rsf)

    if "rsf" not in bg_samples.columns:
        raise KeyError(
            "Sampling the RSF raster did not produce an 'rsf' column."
        )

    background_rsf = bg_samples["rsf"].to_numpy(dtype=float)

    used_rsf = pred.loc[
        pred["used"].astype(bool),
        pred_col,
    ].to_numpy(dtype=float)

    background_rsf = background_rsf[
        np.isfinite(background_rsf) & (background_rsf > 0)
    ]
    used_rsf = used_rsf[
        np.isfinite(used_rsf) & (used_rsf > 0)
    ]

    empty_columns = [
        "bin",
        "quantile_mid",
        "rsf_min",
        "rsf_max",
        "rsf_mid",
        "area_proportion",
        "predicted_mass",
        "observed_n",
        "observed_proportion",
        "expected_n_null",
        "expected_n_model",
        "oe_null",
        "oe_model",
        "pearson_model",
        "oe_model_ci_low",
        "oe_model_ci_high",
    ]

    if background_rsf.size == 0 or used_rsf.size == 0:
        return pd.DataFrame(columns=empty_columns)

    background_log_rsf = np.log(background_rsf)
    used_log_rsf = np.log(used_rsf)

    # Each uniformly sampled background point approximates the same area.
    # Weighting those points by RSF intensity approximates fitted probability
    # mass under uniform baseline accessibility.
    probability_weights = background_rsf / background_rsf.sum()

    # Equal-area quantiles of the available landscape, ordered by RSF score.
    edges = np.quantile(
        background_log_rsf,
        np.linspace(0.0, 1.0, n_bins + 1),
    )

    # Tied raster scores can lead to repeated quantile boundaries.
    edges = np.unique(edges)

    if edges.size < 2:
        return pd.DataFrame(columns=empty_columns)

    # Include held-out values falling marginally outside the sampled range.
    edges[0] = -np.inf
    edges[-1] = np.inf

    actual_n_bins = len(edges) - 1

    background_bin = np.searchsorted(
        edges[1:-1],
        background_log_rsf,
        side="right",
    )

    used_bin = np.searchsorted(
        edges[1:-1],
        used_log_rsf,
        side="right",
    )

    n_used = used_log_rsf.size
    rows = []

    for bin_id in range(actual_n_bins):
        in_background = background_bin == bin_id
        in_used = used_bin == bin_id

        # Proportion of available landscape in this RSF quantile.
        area_proportion = float(in_background.mean())

        # Proportion of fitted static-RSF probability mass in the quantile.
        predicted_mass = float(
            probability_weights[in_background].sum()
        )

        observed_n = int(in_used.sum())
        observed_proportion = observed_n / n_used

        expected_n_null = n_used * area_proportion
        expected_n_model = n_used * predicted_mass

        oe_null = (
            observed_proportion / area_proportion
            if area_proportion > 0
            else np.nan
        )

        oe_model = (
            observed_proportion / predicted_mass
            if predicted_mass > 0
            else np.nan
        )

        model_variance = (
            expected_n_model * (1.0 - predicted_mass)
        )

        pearson_model = (
            (observed_n - expected_n_model)
            / np.sqrt(model_variance)
            if model_variance > 0
            else np.nan
        )

        # Exact marginal interval for the observed bin proportion.
        if observed_n == 0:
            observed_ci_low = 0.0
        else:
            observed_ci_low = beta.ppf(
                alpha / 2,
                observed_n,
                n_used - observed_n + 1,
            )

        if observed_n == n_used:
            observed_ci_high = 1.0
        else:
            observed_ci_high = beta.ppf(
                1.0 - alpha / 2,
                observed_n + 1,
                n_used - observed_n,
            )

        if predicted_mass > 0:
            oe_model_ci_low = observed_ci_low / predicted_mass
            oe_model_ci_high = observed_ci_high / predicted_mass
        else:
            oe_model_ci_low = np.nan
            oe_model_ci_high = np.nan

        scores_in_bin = background_log_rsf[in_background]

        if scores_in_bin.size:
            rsf_min = float(np.min(scores_in_bin))
            rsf_max = float(np.max(scores_in_bin))
            rsf_mid = float(np.mean(scores_in_bin))
        else:
            rsf_min = np.nan
            rsf_max = np.nan
            rsf_mid = np.nan

        rows.append(
            {
                "bin": bin_id + 1,
                "quantile_mid": (
                    bin_id + 0.5
                ) / actual_n_bins,
                "rsf_min": rsf_min,
                "rsf_max": rsf_max,
                "rsf_mid": rsf_mid,
                "area_proportion": area_proportion,
                "predicted_mass": predicted_mass,
                "observed_n": observed_n,
                "observed_proportion": observed_proportion,
                "expected_n_null": expected_n_null,
                "expected_n_model": expected_n_model,
                "oe_null": oe_null,
                "oe_model": oe_model,
                "pearson_model": pearson_model,
                "oe_model_ci_low": oe_model_ci_low,
                "oe_model_ci_high": oe_model_ci_high,
            }
        )

    return pd.DataFrame(rows)

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


def plot_rsf_quantile_calibration(
    calibration_bins: pd.DataFrame,
    *,
    individual_id=None,
    id_col: str = "heldout_ID",
    kind: Literal[
        "proportions",
        "oe",
        "pearson",
    ] = "proportions",
    show_intervals: bool = True,
    annotate_counts: bool = False,
    ax=None,
    figsize: tuple[float, float] = (7, 5),
    title: str | None = None,
):
    """Plot calibration diagnostics across equal-area RSF quantiles.

    Parameters
    ----------
    calibration_bins:
        Output from ``calibration_rsf_quantile_bins``.

    individual_id:
        Optional held-out individual to select when the input contains
        multiple individuals.

    id_col:
        Column identifying held-out individuals.

    kind:
        Diagnostic to display:

        ``"proportions"``
            Compare observed held-out use, fitted static-RSF probability mass,
            and available-area proportion.

        ``"oe"``
            Plot observed / expected use under the fitted static RSF.

        ``"pearson"``
            Plot standardized Pearson residuals.

    show_intervals:
        Show marginal confidence intervals where available. These intervals
        treat relocations as independent and should therefore be interpreted
        cautiously for autocorrelated telemetry.

    annotate_counts:
        Add observed and expected counts to each quantile.

    ax:
        Existing matplotlib axis. A new figure is created when omitted.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    df = calibration_bins.copy()

    # ------------------------------------------------------------
    # Select one held-out individual
    # ------------------------------------------------------------
    selected_id = individual_id

    if individual_id is not None:
        if id_col not in df.columns:
            raise KeyError(
                f"{id_col!r} is not present in calibration_bins."
            )

        df = df.loc[df[id_col] == individual_id].copy()

        if df.empty:
            available_ids = calibration_bins[id_col].dropna().unique()
            raise ValueError(
                f"No calibration results found for "
                f"{individual_id!r}. Available IDs: "
                f"{list(available_ids)}"
            )

    elif id_col in df.columns:
        ids = df[id_col].dropna().unique()

        if len(ids) > 1:
            raise ValueError(
                "calibration_bins contains multiple held-out individuals. "
                "Supply individual_id=... to plot one individual."
            )

        if len(ids) == 1:
            selected_id = ids[0]

    # ------------------------------------------------------------
    # Validate and sort
    # ------------------------------------------------------------
    common_required = {
        "bin",
        "observed_n",
        "observed_proportion",
        "predicted_mass",
        "area_proportion",
        "expected_n_model",
    }

    missing = common_required.difference(df.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .sort_values("bin")
        .copy()
    )

    if df.empty:
        raise ValueError("No calibration results are available to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = df["bin"].to_numpy(dtype=int)

    default_titles = {
        "proportions": "Observed and predicted use across RSF quantiles",
        "oe": "Observed use relative to the fitted static RSF",
        "pearson": "Standardized discrepancy across RSF quantiles",
    }

    if title is None:
        title = default_titles[kind]

        if selected_id is not None:
            title = f"{selected_id} · {title}"

    # ------------------------------------------------------------
    # 1. Direct proportion comparison
    # ------------------------------------------------------------
    if kind == "proportions":
        observed = df["observed_proportion"].to_numpy(dtype=float)
        predicted = df["predicted_mass"].to_numpy(dtype=float)
        available = df["area_proportion"].to_numpy(dtype=float)

        ax.plot(
            x,
            observed,
            marker="o",
            linewidth=1.5,
            label="Observed held-out use",
        )

        ax.plot(
            x,
            predicted,
            marker="s",
            linewidth=1.5,
            label="Predicted by static RSF",
        )

        ax.plot(
            x,
            available,
            linestyle="--",
            linewidth=1.2,
            label="Available area",
        )

        # Convert the O/E confidence interval back to an interval for the
        # observed proportion:
        #
        #     O/E = observed proportion / predicted mass
        #
        interval_cols = {
            "oe_model_ci_low",
            "oe_model_ci_high",
        }

        if show_intervals and interval_cols.issubset(df.columns):
            observed_low = (
                df["oe_model_ci_low"].to_numpy(dtype=float)
                * predicted
            )
            observed_high = (
                df["oe_model_ci_high"].to_numpy(dtype=float)
                * predicted
            )

            valid = (
                np.isfinite(observed_low)
                & np.isfinite(observed_high)
                & (observed_low <= observed)
                & (observed_high >= observed)
            )

            if valid.any():
                ax.errorbar(
                    x[valid],
                    observed[valid],
                    yerr=np.vstack(
                        [
                            observed[valid] - observed_low[valid],
                            observed_high[valid] - observed[valid],
                        ]
                    ),
                    fmt="none",
                    linewidth=0.9,
                    capsize=3,
                )

        ax.set_ylabel("Proportion")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.legend()

    # ------------------------------------------------------------
    # 2. Observed / expected ratio
    # ------------------------------------------------------------
    elif kind == "oe":
        required = {
            "oe_model",
            "oe_model_ci_low",
            "oe_model_ci_high",
        }

        missing = required.difference(df.columns)

        if missing:
            raise KeyError(
                f"The O/E plot requires columns: {sorted(required)}. "
                f"Missing: {sorted(missing)}"
            )

        oe = df["oe_model"].to_numpy(dtype=float)

        ax.plot(
            x,
            oe,
            marker="o",
            linewidth=1.5,
        )

        if show_intervals:
            lower = df["oe_model_ci_low"].to_numpy(dtype=float)
            upper = df["oe_model_ci_high"].to_numpy(dtype=float)

            valid = (
                np.isfinite(lower)
                & np.isfinite(upper)
                & np.isfinite(oe)
                & (lower <= oe)
                & (upper >= oe)
            )

            if valid.any():
                ax.errorbar(
                    x[valid],
                    oe[valid],
                    yerr=np.vstack(
                        [
                            oe[valid] - lower[valid],
                            upper[valid] - oe[valid],
                        ]
                    ),
                    fmt="none",
                    linewidth=0.9,
                    capsize=3,
                )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_ylabel(
            "Observed / expected under fitted static RSF"
        )

    # ------------------------------------------------------------
    # 3. Pearson residuals
    # ------------------------------------------------------------
    elif kind == "pearson":
        if "pearson_model" not in df.columns:
            raise KeyError(
                "The Pearson plot requires a 'pearson_model' column."
            )

        residuals = df["pearson_model"].to_numpy(dtype=float)

        ax.bar(
            x,
            residuals,
        )

        ax.axhline(
            0.0,
            linewidth=1.2,
        )

        # Descriptive reference values only. They are not formal thresholds
        # when telemetry observations remain autocorrelated.
        ax.axhline(
            2.0,
            linestyle=":",
            linewidth=1,
        )

        ax.axhline(
            -2.0,
            linestyle=":",
            linewidth=1,
        )

        ax.set_ylabel("Pearson residual")

    else:
        raise ValueError(
            "kind must be one of "
            "{'proportions', 'oe', 'pearson'}."
        )

    # ------------------------------------------------------------
    # Optional count annotations
    # ------------------------------------------------------------
    if annotate_counts:
        observed_n = df["observed_n"].to_numpy(dtype=int)
        expected_n = df["expected_n_model"].to_numpy(dtype=float)

        if kind == "proportions":
            annotation_y = df[
                "observed_proportion"
            ].to_numpy(dtype=float)

        elif kind == "oe":
            annotation_y = df[
                "oe_model"
            ].to_numpy(dtype=float)

        else:
            annotation_y = df[
                "pearson_model"
            ].to_numpy(dtype=float)

        for xi, yi, observed_i, expected_i in zip(
            x,
            annotation_y,
            observed_n,
            expected_n,
        ):
            if not np.isfinite(yi):
                continue

            ax.annotate(
                f"O={observed_i}\nE={expected_i:.1f}",
                xy=(xi, yi),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i}" for i in x])

    ax.set_xlabel(
        "Equal-area RSF-score quantile (low → high)"
    )
    ax.set_title(title)

    ax.grid(
        axis="y",
        linewidth=0.5,
        alpha=0.25,
    )

    fig.tight_layout()
    return fig, ax