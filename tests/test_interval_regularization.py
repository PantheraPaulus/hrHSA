import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from hsa.movement.kernel import fit_movement_kernel_per_id
from hsa.movement.regularization import regularize_trajectory_interval


def _trajectory(minutes, *, animal="A"):
    start = pd.Timestamp("2025-01-01 00:00:00", tz="Africa/Windhoek")
    rows = []
    for i, minute in enumerate(minutes):
        rows.append(
            {
                "ID": animal,
                "Timestamp": start + pd.Timedelta(minutes=minute),
                "geometry": Point(100.0 * i, 20.0 * (i % 3)),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32733")


def _elapsed_minutes(df):
    start = df["Timestamp"].iloc[0]
    return (
        (df["Timestamp"] - start)
        .dt.total_seconds()
        .div(60)
        .astype(int)
        .tolist()
    )


def test_hourly_data_are_downsampled_to_two_hour_cadence():
    reloc = _trajectory([0, 60, 120, 180, 240, 300, 360])

    regularized = regularize_trajectory_interval(
        reloc,
        id_col="ID",
        timestamp_col="Timestamp",
        expected_interval_min=120,
        tolerance_min=10,
    )

    assert _elapsed_minutes(regularized) == [0, 120, 240, 360]


def test_native_two_hour_data_are_retained():
    reloc = _trajectory([0, 120, 240, 360, 480])

    regularized = regularize_trajectory_interval(
        reloc,
        id_col="ID",
        timestamp_col="Timestamp",
        expected_interval_min=120,
        tolerance_min=10,
    )

    assert _elapsed_minutes(regularized) == [0, 120, 240, 360, 480]


def test_closest_fix_inside_target_window_is_selected():
    reloc = _trajectory([0, 60, 118, 121, 181, 241])

    regularized = regularize_trajectory_interval(
        reloc,
        id_col="ID",
        timestamp_col="Timestamp",
        expected_interval_min=120,
        tolerance_min=10,
    )

    # 121 min is closer to the 120-min target than 118 min. The next target
    # from 121 min is 241 min, which is observed exactly.
    assert _elapsed_minutes(regularized) == [0, 121, 241]


def test_missing_target_is_not_bridged_and_sequence_restarts():
    reloc = _trajectory([0, 60, 180, 240, 300, 420])

    regularized = regularize_trajectory_interval(
        reloc,
        id_col="ID",
        timestamp_col="Timestamp",
        expected_interval_min=120,
        tolerance_min=10,
    )

    # There is no fix around minute 120. Minute 180 becomes a new anchor,
    # producing a valid 180 -> 300 step rather than a false 0 -> 180 step.
    assert _elapsed_minutes(regularized) == [0, 180, 300, 420]

    dt = (
        regularized["Timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
        .to_numpy()
    )
    np.testing.assert_array_equal(dt, np.array([180.0, 120.0, 120.0]))


def test_kernel_constructs_two_hour_steps_from_hourly_relocations():
    reloc = _trajectory(
        [0, 60, 120, 180, 240, 300, 360, 420, 480]
    )

    # Leave round_freq at its default ("h"). The requested analysis cadence
    # must now take precedence and perform the downsampling itself.
    result = fit_movement_kernel_per_id(
        reloc,
        id_col="ID",
        timestamp_col="Timestamp",
        expected_interval_min=120,
        tolerance_min=10,
    )

    regularized = result["reloc_gdf"]
    steps = result["step_df"]

    assert _elapsed_minutes(regularized) == [0, 120, 240, 360, 480]
    assert len(steps) == 4
    assert np.allclose(steps["t_diff_min"], 120.0)


def test_kernel_preserves_subhourly_fixes_until_target_selection():
    reloc = _trajectory(
        [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    )

    result = fit_movement_kernel_per_id(
        reloc,
        id_col="ID",
        timestamp_col="Timestamp",
        expected_interval_min=90,
        tolerance_min=5,
    )

    regularized = result["reloc_gdf"]
    steps = result["step_df"]

    assert _elapsed_minutes(regularized) == [0, 90, 180, 270, 360]
    assert len(steps) == 4
    assert np.allclose(steps["t_diff_min"], 90.0)


def test_regularization_is_applied_per_individual():
    a = _trajectory([0, 60, 120, 180, 240], animal="A")
    b = _trajectory([30, 90, 150, 210, 270], animal="B")
    reloc = pd.concat([a, b], ignore_index=True)
    reloc = gpd.GeoDataFrame(reloc, geometry="geometry", crs=a.crs)

    regularized = regularize_trajectory_interval(
        reloc,
        id_col="ID",
        timestamp_col="Timestamp",
        expected_interval_min=120,
        tolerance_min=10,
    )

    counts = regularized.groupby("ID").size().to_dict()
    assert counts == {"A": 3, "B": 3}

    for _, group in regularized.groupby("ID"):
        dt = (
            group["Timestamp"]
            .diff()
            .dt.total_seconds()
            .div(60)
            .dropna()
        )
        assert np.allclose(dt, 120.0)
