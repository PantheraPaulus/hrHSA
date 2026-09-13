from __future__ import annotations

import pandas as pd

from hsa.movement.distributions import fit_step_distribution, fit_turn_angle_distribution
from hsa.movement.geometry import build_step_data, build_turn_angle_data, prepare_trajectory_data
from hsa.movement.regularization import regularize_trajectory_interval


def fit_movement_kernel_per_id(
    reloc_gdf,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    round_freq: str | None = "h",
    drop_duplicate_fixes: bool = True,
    expected_interval_min: float | None = None,
    tolerance_min: float = 2,
    step_cutoff: float = float("inf"),
) -> dict:
    """Fit step-length and turn-angle distributions per individual.

    ``reloc_gdf`` must already be a projected GeoDataFrame. CRS conversion is
    intentionally left outside this function so callers make the spatial unit
    explicit.

    When ``expected_interval_min`` is supplied, trajectories are regularized to
    that cadence *before* step lengths and turning angles are calculated. Thus
    finer-resolution data are downsampled to the requested analysis interval
    rather than having their shorter adjacent steps discarded after the fact.
    Missing target fixes break/restart a sequence and are never bridged or
    interpolated. The interval filters in ``build_step_data`` and
    ``build_turn_angle_data`` remain as final safety checks.

    Target-interval regularization supersedes ``round_freq``-based temporal
    deduplication. This preserves sub-hourly observations long enough to select
    the fix closest to each requested target (for example 30-min data requested
    at a 90-min analysis interval). Exact duplicate timestamps are still
    reduced to one relocation per individual before regularization.
    """

    regularize = expected_interval_min is not None

    reloc = prepare_trajectory_data(
        reloc_gdf,
        id_col=id_col,
        timestamp_col=timestamp_col,
        round_freq=None if regularize else round_freq,
        drop_duplicate_fixes=False if regularize else drop_duplicate_fixes,
    )
    reloc = reloc_gdf.loc[reloc.index].copy() if "geometry" not in reloc.columns else reloc

    if regularize:
        if drop_duplicate_fixes:
            reloc = (
                reloc.drop_duplicates(
                    [id_col, timestamp_col],
                    keep="first",
                )
                .sort_values([id_col, timestamp_col])
                .reset_index(drop=True)
            )

        reloc = regularize_trajectory_interval(
            reloc,
            id_col=id_col,
            timestamp_col=timestamp_col,
            expected_interval_min=expected_interval_min,
            tolerance_min=tolerance_min,
        )

    step_df = build_step_data(
        reloc,
        id_col=id_col,
        timestamp_col=timestamp_col,
        expected_interval_min=expected_interval_min,
        tolerance_min=tolerance_min,
    )
    angle_df = build_turn_angle_data(
        step_df,
        id_col=id_col,
        timestamp_col=timestamp_col,
        expected_interval_min=expected_interval_min,
        tolerance_min=tolerance_min,
    )

    rows = []
    for animal_id in step_df[id_col].dropna().unique():
        steps = step_df.loc[step_df[id_col] == animal_id, "step_m"]
        angles = angle_df.loc[angle_df[id_col] == animal_id, "turn_angle"]
        if steps.dropna().empty or angles.dropna().empty:
            continue

        step_fit = fit_step_distribution(steps, cutoff=step_cutoff)
        angle_fit = fit_turn_angle_distribution(angles)
        rows.append(
            {
                id_col: animal_id,
                "n_steps": step_fit["n"],
                "step_distribution": step_fit["distribution"],
                "step_params": step_fit["params"],
                "step_q25": step_fit["q25"],
                "step_median": step_fit["median"],
                "step_mean": step_fit["mean"],
                "step_q75": step_fit["q75"],
                "step_max": step_fit["max"],
                "n_angles": angle_fit["n"],
                "angle_distribution": angle_fit["distribution"],
                "angle_params": angle_fit["params"],
            }
        )

    return {
        "reloc_gdf": reloc,
        "step_df": step_df,
        "angle_df": angle_df,
        "summary": pd.DataFrame(rows),
    }
