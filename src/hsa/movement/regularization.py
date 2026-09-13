from __future__ import annotations

import numpy as np
import pandas as pd

from hsa._time import require_timezone_aware


def regularize_trajectory_interval(
    df: pd.DataFrame,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    expected_interval_min: float,
    tolerance_min: float = 2,
) -> pd.DataFrame:
    """Regularize trajectories to a requested analysis interval.

    The first relocation for each individual anchors a sequence of target
    times. For every retained relocation, the next retained relocation is the
    observed fix closest to ``expected_interval_min`` minutes later, provided
    it lies within ``tolerance_min`` minutes of that target.

    This means finer-resolution data are genuinely downsampled before movement
    metrics are calculated. For example, an hourly trajectory requested at a
    120-minute interval is reduced from ``00, 01, 02, 03, ...`` to
    ``00, 02, 04, ...`` rather than calculating 60-minute steps and then
    discarding them.

    Missing target fixes are never bridged or interpolated. If no observation
    falls inside a target window, that sequence ends and regularization
    restarts at the first observation after the missed window. Consequently a
    failed GPS fix cannot turn a longer gap into an apparently valid movement
    step.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Relocation records. The input dataframe type and columns are preserved.

    id_col : str, default "Individual_ID"
        Individual identifier column.

    timestamp_col : str, default "Timestamp"
        Timezone-aware timestamp column.

    expected_interval_min : float
        Requested interval between retained relocations, in minutes.

    tolerance_min : float, default 2
        Symmetric tolerance around each target time, in minutes. If more than
        one fix lies inside the window, the fix closest to the exact target is
        retained; ties are resolved in favour of the earlier fix.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        Temporally regularized relocations, sorted by individual and time.

    Notes
    -----
    This function can downsample finer-resolution data but never upsamples or
    interpolates coarser data. A subsequent interval check should therefore be
    retained when constructing movement steps.
    """

    missing = [
        col
        for col in (id_col, timestamp_col)
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not np.isfinite(expected_interval_min) or expected_interval_min <= 0:
        raise ValueError("expected_interval_min must be a finite value > 0.")

    if not np.isfinite(tolerance_min) or tolerance_min < 0:
        raise ValueError("tolerance_min must be a finite value >= 0.")

    if tolerance_min >= expected_interval_min:
        raise ValueError(
            "tolerance_min must be smaller than expected_interval_min."
        )

    g = df.copy()
    g[timestamp_col] = require_timezone_aware(
        g[timestamp_col],
        name=timestamp_col,
    )
    g = (
        g.dropna(subset=[id_col, timestamp_col])
        .sort_values([id_col, timestamp_col])
        .reset_index(drop=True)
    )

    target_interval = pd.Timedelta(minutes=float(expected_interval_min))
    tolerance = pd.Timedelta(minutes=float(tolerance_min))

    selected_indices: list[int] = []

    for _, group in g.groupby(id_col, sort=False):
        group = group.sort_values(timestamp_col)
        if group.empty:
            continue

        indices = group.index.to_numpy()
        times = group[timestamp_col].reset_index(drop=True)

        anchor_pos = 0
        selected_indices.append(int(indices[anchor_pos]))

        while anchor_pos < len(group) - 1:
            anchor_time = times.iloc[anchor_pos]
            target_time = anchor_time + target_interval
            lower = target_time - tolerance
            upper = target_time + tolerance

            future_positions = np.arange(anchor_pos + 1, len(group))
            future_times = times.iloc[future_positions]

            inside_window = (
                (future_times >= lower)
                & (future_times <= upper)
            ).to_numpy()
            candidate_positions = future_positions[inside_window]

            if candidate_positions.size:
                differences_ns = np.array(
                    [
                        abs(times.iloc[pos] - target_time).value
                        for pos in candidate_positions
                    ],
                    dtype=np.int64,
                )
                chosen_pos = int(
                    candidate_positions[np.argmin(differences_ns)]
                )
                selected_indices.append(int(indices[chosen_pos]))
                anchor_pos = chosen_pos
                continue

            # No fix near the requested target. Do not bridge the gap: restart
            # from the first observation beyond the failed target window.
            after_window = future_positions[
                (future_times > upper).to_numpy()
            ]
            if not after_window.size:
                break

            anchor_pos = int(after_window[0])
            selected_indices.append(int(indices[anchor_pos]))

    out = g.loc[selected_indices].copy()
    return (
        out.sort_values([id_col, timestamp_col])
        .reset_index(drop=True)
    )


__all__ = ["regularize_trajectory_interval"]
