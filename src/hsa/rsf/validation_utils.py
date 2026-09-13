"""Shared utilities for RSF validation uncertainty workflows.

These helpers are estimator-independent. Frequentist and Bayesian validation
modules both operate on predicted-selection ranks and real timestamps, so the
common Boyce-curve and temporal-block logic lives here rather than in either
estimator-specific module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hsa._time import require_timezone_aware
from scipy.stats import spearmanr


def boyce_curve_from_ranks(
    ranks: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate an equal-availability P/E curve and Boyce index from ranks."""
    if n_bins < 3:
        raise ValueError("n_bins must be at least 3.")

    values = np.asarray(ranks, dtype=float)
    values = values[np.isfinite(values)]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    if len(values) == 0:
        return midpoints, np.full(n_bins, np.nan), np.nan

    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=edges)
    if counts.sum() == 0:
        return midpoints, np.full(n_bins, np.nan), np.nan

    pe_ratio = counts / counts.sum() * n_bins
    if np.nanstd(pe_ratio) == 0:
        return midpoints, pe_ratio, np.nan

    boyce = float(spearmanr(midpoints, pe_ratio).statistic)
    return midpoints, pe_ratio, boyce


def calendar_temporal_units(
    timestamps,
    *,
    unit: str = "W",
) -> pd.DataFrame:
    """Return sortable temporal units and human-readable calendar labels."""
    ts = pd.Series(
        require_timezone_aware(
            np.asarray(timestamps),
            name="timestamps",
        )
    )
    if ts.isna().any():
        raise ValueError("Missing/invalid timestamps in temporal validation data.")

    ts_naive = ts.dt.tz_localize(None)

    if unit.upper().startswith("W"):
        iso = ts.dt.isocalendar()
        label = (
            iso["year"].astype(str)
            + "-W"
            + iso["week"].astype(str).str.zfill(2)
        )
        temporal_unit = ts_naive.dt.to_period("W-SUN").dt.start_time
    else:
        try:
            period = ts_naive.dt.to_period(unit)
        except Exception as exc:
            raise ValueError(
                f"Could not interpret temporal_unit={unit!r}; use a calendar "
                "period such as 'D', 'W', or 'M'."
            ) from exc
        temporal_unit = period.dt.start_time
        label = period.astype(str)

    return pd.DataFrame(
        {
            "Timestamp": ts,
            "temporal_unit": temporal_unit,
            "temporal_label": label.to_numpy(),
        }
    )


def assign_contiguous_temporal_folds(
    timestamps,
    *,
    k: int,
    unit: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign consecutive whole calendar units to contiguous validation folds."""
    temporal = calendar_temporal_units(
        timestamps,
        unit=unit,
    )
    units = np.array(
        sorted(temporal["temporal_unit"].unique()),
        dtype="datetime64[ns]",
    )
    if len(units) < k:
        raise ValueError(
            f"Only {len(units)} temporal units are available, but k={k} "
            "folds were requested."
        )

    chunks = np.array_split(units, k)
    mapping: dict[pd.Timestamp, int] = {}
    for fold, chunk in enumerate(chunks):
        for value in chunk:
            mapping[pd.Timestamp(value)] = fold

    folds = temporal["temporal_unit"].map(mapping).to_numpy(dtype=int)
    return folds, temporal


def fixed_duration_block_labels(
    timestamps,
    block: str | None,
) -> pd.Series:
    """Assign observations to fixed-duration temporal bootstrap blocks."""
    ts = pd.Series(
        require_timezone_aware(
            np.asarray(timestamps),
            name="timestamps",
        )
    )
    if ts.isna().any():
        raise ValueError("Bootstrap validation requires valid timestamps.")
    if block is None:
        return pd.Series(np.arange(len(ts)), index=ts.index)

    try:
        return ts.dt.floor(block)
    except Exception as exc:
        raise ValueError(
            f"bootstrap_block={block!r} must be a fixed duration such as "
            "'1D', '3D', or '7D'."
        ) from exc


__all__ = [
    "assign_contiguous_temporal_folds",
    "boyce_curve_from_ranks",
    "calendar_temporal_units",
    "fixed_duration_block_labels",
]
