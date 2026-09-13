from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import geopandas as gpd
import pandas as pd

from pathlib import Path
from collections.abc import Iterable

import pandas as pd
import geopandas as gpd
from shapely import wkb, wkt
from shapely.geometry.base import BaseGeometry
import pyarrow.dataset as ds

from hsa._time import require_timezone_aware

def load_trajectory_data(
    folder: str | Path,
    *,
    individuals: Iterable[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    min_relocations: int = 1,
    id_col: str = "individual-local-identifier",
    timestamp_col: str = "timestamp",
    lon_col: str = "location-long",
    lat_col: str = "location-lat",
    filter_rows: bool = True,
) -> pd.DataFrame:
    """
    Load relocations from a Hive-partitioned Parquet dataset.

    Expected folder structure
    -------------------------
    folder/
        individual-local-identifier=BG0195_Argentera/
            *.parquet
        individual-local-identifier=BG0321_Veronika/
            *.parquet
        ...

    Parameters
    ----------
    folder : str or Path
        Root directory of the partitioned Parquet dataset.

    individuals : iterable of str, optional
        Individual IDs to retain, e.g.
        ["BG0195_Argentera", "BG0321_Veronika"].

    start, end : str or pandas.Timestamp, optional
        Temporal selection interval. ``start`` is inclusive,
        ``end`` is exclusive.

    bbox : tuple of float, optional
        Bounding box as::

            (xmin, ymin, xmax, ymax)

        Coordinates must correspond to lon_col / lat_col.

    min_relocations : int, default 1
        Minimum number of relocations within the requested
        temporal/spatial window required for an individual
        to be retained.

    id_col : str
        Hive partition column containing the individual ID.

    timestamp_col : str
        Timestamp column.

    lon_col, lat_col : str
        Longitude and latitude columns.

    filter_rows : bool, default True
        If True, return only relocations inside the requested
        time/spatial window.

        If False, use the window only to identify qualifying
        individuals and then return their complete trajectories.

    Returns
    -------
    pandas.DataFrame
        Relocations belonging to qualifying individuals.
    """

    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Trajectory dataset does not exist: {folder}"
        )

    if min_relocations < 1:
        raise ValueError("min_relocations must be >= 1.")

    # --------------------------------------------------------------
    # Open Hive-partitioned Parquet dataset
    # --------------------------------------------------------------

    dataset = ds.dataset(
        folder,
        format="parquet",
        partitioning="hive",
    )

    available_columns = set(dataset.schema.names)

    required = {
        id_col,
        timestamp_col,
    }

    if bbox is not None:
        required.update({lon_col, lat_col})

    missing = required - available_columns

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------------
    # Build selection expression
    # --------------------------------------------------------------

    expression = None

    def add_filter(new_expression):
        nonlocal expression

        if expression is None:
            expression = new_expression
        else:
            expression = expression & new_expression

    # Individual partition filter
    if individuals is not None:

        individuals = [
            str(x)
            for x in individuals
        ]

        if not individuals:
            raise ValueError(
                "individuals was supplied but is empty."
            )

        add_filter(
            ds.field(id_col).isin(individuals)
        )

    # --------------------------------------------------------------
    # Temporal filters
    # --------------------------------------------------------------

    start_ts = (
        pd.Timestamp(start)
        if start is not None
        else None
    )

    end_ts = (
        pd.Timestamp(end)
        if end is not None
        else None
    )

    if (
        start_ts is not None
        and end_ts is not None
        and start_ts >= end_ts
    ):
        raise ValueError(
            "start must be earlier than end."
        )

    if start_ts is not None:
        add_filter(
            ds.field(timestamp_col)
            >= start_ts.to_pydatetime()
        )

    if end_ts is not None:
        add_filter(
            ds.field(timestamp_col)
            < end_ts.to_pydatetime()
        )

    # --------------------------------------------------------------
    # Spatial bounding-box filter
    # --------------------------------------------------------------

    if bbox is not None:

        xmin, ymin, xmax, ymax = bbox

        if xmin > xmax or ymin > ymax:
            raise ValueError(
                "Invalid bbox: expected "
                "(xmin, ymin, xmax, ymax)."
            )

        add_filter(
            (ds.field(lon_col) >= xmin)
            & (ds.field(lon_col) <= xmax)
            & (ds.field(lat_col) >= ymin)
            & (ds.field(lat_col) <= ymax)
        )

    # --------------------------------------------------------------
    # Read only records matching the candidate window
    # --------------------------------------------------------------

    candidate_table = dataset.to_table(
        filter=expression,
    )

    candidate = candidate_table.to_pandas()

    if candidate.empty:
        raise ValueError(
            "No relocations match the requested selection."
        )

    # --------------------------------------------------------------
    # Apply minimum relocation criterion per individual
    # --------------------------------------------------------------

    counts = (
        candidate
        .groupby(id_col)
        .size()
    )

    qualifying_ids = counts[
        counts >= min_relocations
    ].index.tolist()

    if not qualifying_ids:
        raise ValueError(
            "No individuals satisfy "
            f"min_relocations={min_relocations} "
            "within the requested window."
        )

    # --------------------------------------------------------------
    # Return requested rows
    # --------------------------------------------------------------

    if filter_rows:

        # Candidate data already contain exactly the requested
        # temporal/spatial subset.
        result = candidate[
            candidate[id_col].isin(qualifying_ids)
        ].copy()

    else:

        # Window was only used to select individuals.
        # Now retrieve complete trajectories for those individuals.
        full_expression = (
            ds.field(id_col)
            .isin(qualifying_ids)
        )

        result = (
            dataset
            .to_table(filter=full_expression)
            .to_pandas()
        )

    # --------------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------------

    result = result.reset_index(drop=True)

    return result

def prepare_trajectory_data(
    df: pd.DataFrame,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    geometry_col: str | None = "geometry",
    lon_col: str | None = None,
    lat_col: str | None = None,
    source_crs: str | int = "EPSG:4326",
    target_crs: str | int | None = None,
    round_freq: str | None = "h",
    drop_duplicate_fixes: bool = True,
) -> gpd.GeoDataFrame:
    """
    Prepare relocation records as a GeoDataFrame for movement analysis.

    Supports:
    - existing GeoDataFrames,
    - Shapely geometry,
    - WKB geometry,
    - WKT geometry,
    - longitude/latitude columns.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Input relocation data.

    id_col : str
        Individual identifier column.

    timestamp_col : str
        Timestamp column.

    geometry_col : str or None
        Geometry column. It may contain Shapely objects, WKB, or WKT.

    lon_col, lat_col : str or None
        Coordinate columns used when no geometry column is available.

    source_crs : str or int
        CRS assumed if CRS metadata are unavailable.

    target_crs : str or int or None
        Target CRS. If None, coordinates are not reprojected.

    round_freq : str or None
        Pandas frequency used to create ``time_rounded``.
        For example ``"h"`` for hourly intervals.
        If None, no temporal rounding is performed.

    drop_duplicate_fixes : bool
        If True, retain at most one relocation per individual and
        rounded time interval.

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned trajectory data.
    """

    g = df.copy()

    # --------------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------------

    missing = [
        col
        for col in (id_col, timestamp_col)
        if col not in g.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------

    if (
        isinstance(g, gpd.GeoDataFrame)
        and geometry_col is not None
        and geometry_col in g.columns
    ):
        # GeoParquet read with geopandas already gives us geometry.
        if g.geometry.name != geometry_col:
            g = g.set_geometry(geometry_col)

    elif geometry_col is not None and geometry_col in g.columns:

        def parse_geometry(value):

            if value is None:
                return None

            if isinstance(value, BaseGeometry):
                return value

            if isinstance(
                value,
                (bytes, bytearray, memoryview),
            ):
                return wkb.loads(bytes(value))

            if isinstance(value, str):
                if not value.strip():
                    return None

                return wkt.loads(value)

            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass

            raise TypeError(
                "Unsupported geometry representation "
                f"{type(value).__name__!r}."
            )

        g[geometry_col] = g[geometry_col].map(
            parse_geometry
        )

        input_crs = (
            getattr(df, "crs", None)
            or source_crs
        )

        g = gpd.GeoDataFrame(
            g,
            geometry=geometry_col,
            crs=input_crs,
        )

    elif lon_col is not None and lat_col is not None:

        missing_xy = [
            col
            for col in (lon_col, lat_col)
            if col not in g.columns
        ]

        if missing_xy:
            raise ValueError(
                f"Missing coordinate columns: {missing_xy}"
            )

        # Explicit numeric conversion is useful for CSV-style input.
        g[lon_col] = pd.to_numeric(
            g[lon_col],
            errors="coerce",
        )

        g[lat_col] = pd.to_numeric(
            g[lat_col],
            errors="coerce",
        )

        g = g.dropna(
            subset=[lon_col, lat_col]
        )

        g = gpd.GeoDataFrame(
            g,
            geometry=gpd.points_from_xy(
                g[lon_col],
                g[lat_col],
            ),
            crs=source_crs,
        )

    else:
        raise ValueError(
            "Provide either a valid geometry_col or "
            "both lon_col and lat_col."
        )

    # --------------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------------

    timestamps = require_timezone_aware(
        g[timestamp_col],
        name=timestamp_col,
    )
    g[timestamp_col] = timestamps
    g["Timestamp"] = timestamps

    g = g.dropna(
        subset=[
            id_col,
            "Timestamp",
            g.geometry.name,
        ]
    )

    # --------------------------------------------------------------
    # CRS
    # --------------------------------------------------------------

    if g.crs is None:
        g = g.set_crs(source_crs)

    if target_crs is not None:
        g = g.to_crs(target_crs)

    # --------------------------------------------------------------
    # Temporal regularisation
    # --------------------------------------------------------------

    if round_freq is not None:

        g["time_rounded"] = (
            g[timestamp_col]
            .dt.floor(round_freq)
        )

        if drop_duplicate_fixes:
            g = g.drop_duplicates(
                [id_col, "time_rounded"],
                keep="first",
            )

    # --------------------------------------------------------------
    # Final ordering
    # --------------------------------------------------------------

    g = (
        g
        .sort_values(
            [id_col, timestamp_col]
        )
        .reset_index(drop=True)
    )

    # Lightweight metadata
    g.attrs["source_crs"] = str(source_crs)

    g.attrs["target_crs"] = (
        str(target_crs)
        if target_crs is not None
        else None
    )

    return g

def build_step_data(
    reloc_gdf,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    expected_interval_min: float | None = None,
    tolerance_min: float = 2,
):
    """Calculate step lengths, time differences, and speeds from a projected GeoDataFrame."""

    g = reloc_gdf.sort_values([id_col, timestamp_col]).reset_index(drop=True).copy()
    g["previous_timestamp"] = g.groupby(id_col)[timestamp_col].shift(1)
    g["previous_location"] = g.groupby(id_col)["geometry"].shift(1)
    g["t_diff_h"] = (g[timestamp_col] - g["previous_timestamp"]).dt.total_seconds() / 3600.0
    g["step_m"] = g.geometry.distance(g["previous_location"])
    g = g.loc[g["t_diff_h"].notna() & (g["t_diff_h"] > 0)].copy()
    g["speed_kmh"] = (g["step_m"] / 1000.0) / g["t_diff_h"]
    g["t_diff_min"] = g["t_diff_h"] * 60.0

    if expected_interval_min is not None:
        lower = expected_interval_min - tolerance_min
        upper = expected_interval_min + tolerance_min
        g = g.loc[g["t_diff_min"].between(lower, upper, inclusive="both")].copy()

    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["step_m", "t_diff_h"])
    g = g.loc[g["step_m"] > 0].copy()
    if g.empty:
        raise ValueError("No valid steps remaining after interval filtering.")
    return g


def build_displacement_velocity_data(
    reloc_gdf,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    expected_interval_min: float | None = None,
    tolerance_min: float = 2,
):
    """Build a step table with displacement and velocity vector components.

    The input must be a projected GeoDataFrame, so x/y units are metric. Each
    returned row represents the movement from the previous fix to the current
    fix for the same individual.

    Added columns include:
    - ``x``, ``y``: current coordinates
    - ``x_prev``, ``y_prev``: previous coordinates
    - ``dx_m``, ``dy_m``: displacement vector components in metres
    - ``vx_m_per_h``, ``vy_m_per_h``: velocity vector components
    - ``speed_m_per_h``: scalar speed
    - ``heading``: movement direction in radians
    """

    g = build_step_data(
        reloc_gdf,
        id_col=id_col,
        timestamp_col=timestamp_col,
        expected_interval_min=expected_interval_min,
        tolerance_min=tolerance_min,
    ).copy()

    g["x"] = g.geometry.x
    g["y"] = g.geometry.y
    g["x_prev"] = g["previous_location"].x
    g["y_prev"] = g["previous_location"].y

    g["dx_m"] = g["x"] - g["x_prev"]
    g["dy_m"] = g["y"] - g["y_prev"]
    g["vx_m_per_h"] = g["dx_m"] / g["t_diff_h"]
    g["vy_m_per_h"] = g["dy_m"] / g["t_diff_h"]
    g["speed_m_per_h"] = np.hypot(g["vx_m_per_h"], g["vy_m_per_h"])
    g["heading"] = np.arctan2(g["dy_m"], g["dx_m"])

    g = g.replace([np.inf, -np.inf], np.nan)
    g = g.dropna(subset=["dx_m", "dy_m", "vx_m_per_h", "vy_m_per_h", "heading"])
    if g.empty:
        raise ValueError("No valid displacement/velocity vectors available.")
    return g


def build_lagged_vector_pairs(
    vector_df: pd.DataFrame,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    vector_cols: tuple[str, str] = ("vx_m_per_h", "vy_m_per_h"),
    max_lag: str | pd.Timedelta = "48h",
    lag_bin: str | pd.Timedelta = "1h",
) -> pd.DataFrame:
    """Create lagged vector pairs for empirical autocorrelation estimation.

    Each returned row pairs one movement vector with a later movement vector
    from the same individual. The dot product and cosine similarity are included
    so downstream code can estimate how vector similarity declines with time lag.

    This is intentionally a dataframe builder, not the final decorrelation-time
    estimator. It belongs in ``geometry`` because it is purely geometric and
    temporal preprocessing.
    """

    required = [id_col, timestamp_col, *vector_cols]
    missing = [col for col in required if col not in vector_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    max_lag_td = pd.Timedelta(max_lag)
    lag_bin_td = pd.Timedelta(lag_bin)
    if max_lag_td <= pd.Timedelta(0):
        raise ValueError("max_lag must be positive.")
    if lag_bin_td <= pd.Timedelta(0):
        raise ValueError("lag_bin must be positive.")

    vx, vy = vector_cols
    rows = []
    for animal_id, group in vector_df.sort_values([id_col, timestamp_col]).groupby(id_col):
        g = group.reset_index(drop=True).copy()
        times = pd.to_datetime(g[timestamp_col]).to_numpy(dtype="datetime64[ns]")
        vec = g[[vx, vy]].to_numpy(dtype=float)
        norms = np.linalg.norm(vec, axis=1)

        for i in range(len(g) - 1):
            dt = pd.to_timedelta(times[i + 1 :] - times[i])
            within = np.where(dt <= max_lag_td)[0]
            if within.size == 0:
                continue

            for offset in within:
                j = i + 1 + int(offset)
                norm_product = norms[i] * norms[j]
                dot = float(np.dot(vec[i], vec[j]))
                cosine = dot / norm_product if norm_product > 0 else np.nan
                lag = pd.Timedelta(times[j] - times[i])
                lag_bin_value = pd.to_timedelta(
                    np.floor(lag / lag_bin_td) * lag_bin_td,
                    unit="ns",
                )

                rows.append(
                    {
                        id_col: animal_id,
                        "t0": pd.Timestamp(times[i]),
                        "t1": pd.Timestamp(times[j]),
                        "lag": lag,
                        "lag_h": lag.total_seconds() / 3600.0,
                        "lag_bin": lag_bin_value,
                        "lag_bin_h": lag_bin_value.total_seconds() / 3600.0,
                        f"{vx}_0": vec[i, 0],
                        f"{vy}_0": vec[i, 1],
                        f"{vx}_1": vec[j, 0],
                        f"{vy}_1": vec[j, 1],
                        "dot_product": dot,
                        "cosine_similarity": cosine,
                        "speed_0": norms[i],
                        "speed_1": norms[j],
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No lagged vector pairs found. Increase max_lag or check timestamps.")
    return out


def build_turn_angle_data(
    step_df,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    expected_interval_min: float | None = None,
    tolerance_min: float = 2,
):
    """Calculate turning angles from consecutive steps."""

    g = step_df.sort_values([id_col, timestamp_col]).reset_index(drop=True).copy()
    valid_ids = g.groupby(id_col).size().loc[lambda s: s >= 3].index
    g = g.loc[g[id_col].isin(valid_ids)].copy()
    if g.empty:
        raise ValueError("No tracks with at least 3 fixes available for turning-angle calculation.")

    g["next_timestamp"] = g.groupby(id_col)[timestamp_col].shift(-1)
    g["next_position"] = g.groupby(id_col)["geometry"].shift(-1)
    g["t_diff_next_min"] = (g["next_timestamp"] - g[timestamp_col]).dt.total_seconds() / 60.0

    if expected_interval_min is not None:
        lower = expected_interval_min - tolerance_min
        upper = expected_interval_min + tolerance_min
        g = g.loc[g["t_diff_next_min"].between(lower, upper, inclusive="both")].copy()

    g["x"] = g.geometry.x
    g["y"] = g.geometry.y
    g["x_prev"] = g["previous_location"].x
    g["y_prev"] = g["previous_location"].y
    g["x_next"] = g["next_position"].x
    g["y_next"] = g["next_position"].y

    heading_in = np.arctan2(g["y"] - g["y_prev"], g["x"] - g["x_prev"])
    heading_out = np.arctan2(g["y_next"] - g["y"], g["x_next"] - g["x"])
    g["turn_angle"] = (heading_out - heading_in + np.pi) % (2 * np.pi) - np.pi

    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["turn_angle"])
    if g.empty:
        raise ValueError("No valid turning angles remaining after filtering.")
    return g

def thin_trajectory_data(
    df: pd.DataFrame,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    min_interval: str | pd.Timedelta = "1h",
    keep: str = "first",
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Temporally thin trajectory data while enforcing a minimum interval
    between retained relocations.

    Unlike clock-based temporal binning, this function guarantees that
    consecutive retained relocations for each individual are separated
    by at least ``min_interval``.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Relocation data.

    id_col : str, default "Individual_ID"
        Individual identifier column.

    timestamp_col : str, default "Timestamp"
        Timestamp column.

    min_interval : str or pandas.Timedelta, default "1h"
        Minimum allowed time interval between retained relocations.
        Examples: ``"1h"``, ``"30min"``, ``"2h"``.

    keep : {"first", "last", "random"}, default "first"
        Strategy used to thin each individual's trajectory.

        ``"first"``
            Start with the earliest relocation and retain the next
            relocation occurring at least ``min_interval`` later.

        ``"last"``
            Start with the latest relocation and work backwards,
            retaining the preceding relocation occurring at least
            ``min_interval`` earlier.

        ``"random"``
            Consider relocations in random order and retain a relocation
            only if it is at least ``min_interval`` from all already
            retained relocations. This produces a random, reproducible
            temporally thinned subset.

    random_state : int or None, optional
        Random seed used when ``keep="random"``.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        Temporally thinned trajectory data. The input dataframe type
        and columns are preserved.

    Notes
    -----
    ``keep="first"`` and ``keep="last"`` are deterministic.

    ``keep="random"`` does not correspond to fixed clock-hour bins.
    Instead, it randomly selects relocations subject to the same strict
    minimum-separation constraint.
    """

    # --------------------------------------------------------------
    # Validate input
    # --------------------------------------------------------------

    missing = [
        col
        for col in (id_col, timestamp_col)
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if keep not in {"first", "last", "random"}:
        raise ValueError(
            "keep must be one of: "
            "'first', 'last', or 'random'."
        )

    interval = pd.Timedelta(min_interval)

    if interval <= pd.Timedelta(0):
        raise ValueError(
            "min_interval must be greater than zero."
        )

    # --------------------------------------------------------------
    # Prepare timestamps
    # --------------------------------------------------------------

    g = df.copy()

    g[timestamp_col] = pd.to_datetime(
        g[timestamp_col],
        errors="coerce",
    )

    g = g.dropna(
        subset=[id_col, timestamp_col]
    )

    g = g.sort_values(
        [id_col, timestamp_col]
    )

    rng = np.random.default_rng(random_state)

    selected_indices = []

    # --------------------------------------------------------------
    # Thin each individual separately
    # --------------------------------------------------------------

    for _, group in g.groupby(id_col, sort=False):

        times = group[timestamp_col]
        indices = group.index.to_numpy()

        # ----------------------------------------------------------
        # Keep first
        # ----------------------------------------------------------

        if keep == "first":

            chosen = [0]
            last_time = times.iloc[0]

            for pos in range(1, len(group)):

                current_time = times.iloc[pos]

                if current_time - last_time >= interval:
                    chosen.append(pos)
                    last_time = current_time

            selected_indices.extend(
                indices[chosen]
            )

        # ----------------------------------------------------------
        # Keep last
        # ----------------------------------------------------------

        elif keep == "last":

            chosen = [len(group) - 1]
            last_time = times.iloc[-1]

            for pos in range(
                len(group) - 2,
                -1,
                -1,
            ):

                current_time = times.iloc[pos]

                if last_time - current_time >= interval:
                    chosen.append(pos)
                    last_time = current_time

            selected_indices.extend(
                indices[chosen]
            )

        # ----------------------------------------------------------
        # Random
        # ----------------------------------------------------------

        else:

            order = rng.permutation(len(group))

            chosen = []

            times_ns = (
                times
                .astype("int64")
                .to_numpy()
            )

            interval_ns = interval.value

            for pos in order:

                current_time = times_ns[pos]

                if not chosen:
                    chosen.append(pos)
                    continue

                chosen_times = times_ns[chosen]

                sufficiently_far = np.all(
                    np.abs(
                        chosen_times - current_time
                    )
                    >= interval_ns
                )

                if sufficiently_far:
                    chosen.append(pos)

            selected_indices.extend(
                indices[chosen]
            )

    # --------------------------------------------------------------
    # Return in chronological order
    # --------------------------------------------------------------

    result = (
        g.loc[selected_indices]
        .sort_values(
            [id_col, timestamp_col]
        )
        .reset_index(drop=True)
    )

    return result