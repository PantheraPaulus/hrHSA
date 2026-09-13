import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from hsa._time import require_timezone_aware
from hsa.movement.geometry import prepare_trajectory_data
from hsa.rsf.validation_utils import calendar_temporal_units
from hsa.ssf import FrequentistISSF


def _reloc(timestamps):
    return gpd.GeoDataFrame(
        {
            "ID": ["A"] * len(timestamps),
            "Timestamp": timestamps,
        },
        geometry=[Point(i, 0) for i in range(len(timestamps))],
        crs="EPSG:3857",
    )


def test_require_timezone_aware_rejects_naive_timestamps():
    values = pd.Series(pd.date_range("2026-01-01 18:00", periods=3, freq="h"))
    with pytest.raises(ValueError, match="timezone-aware"):
        require_timezone_aware(values)


def test_require_timezone_aware_preserves_local_clock_and_converts_only_on_request():
    values = pd.Series(
        pd.date_range(
            "2026-01-01 18:00",
            periods=2,
            freq="h",
            tz="Africa/Windhoek",
        )
    )
    local = require_timezone_aware(values)
    utc = require_timezone_aware(values, to_utc=True)

    assert str(local.dt.tz) == "Africa/Windhoek"
    assert local.iloc[0].hour == 18
    assert str(utc.dt.tz) == "UTC"
    assert utc.iloc[0].hour == 16


def test_frequentist_issf_rejects_naive_relocations():
    reloc = _reloc(pd.date_range("2026-01-01 18:00", periods=3, freq="h"))
    with pytest.raises(ValueError, match="timezone-aware"):
        FrequentistISSF(reloc, id_col="ID", timestamp_col="Timestamp")


def test_frequentist_issf_preserves_supplied_timezone():
    reloc = _reloc(
        pd.date_range(
            "2026-01-01 18:00",
            periods=3,
            freq="h",
            tz="Africa/Windhoek",
        )
    )
    analysis = FrequentistISSF(reloc, id_col="ID", timestamp_col="Timestamp")

    assert str(analysis.reloc["Timestamp"].dt.tz) == "Africa/Windhoek"
    assert analysis.reloc["Timestamp"].iloc[0].hour == 18


def test_prepare_trajectory_data_rejects_naive_timestamps():
    reloc = _reloc(pd.date_range("2026-01-01 18:00", periods=3, freq="h"))
    with pytest.raises(ValueError, match="timezone-aware"):
        prepare_trajectory_data(reloc, id_col="ID", timestamp_col="Timestamp")


def test_calendar_units_use_local_not_utc_date():
    local = pd.DatetimeIndex([pd.Timestamp("2026-08-31 00:30", tz="Africa/Windhoek")])
    temporal = calendar_temporal_units(local, unit="D")
    assert temporal.loc[0, "temporal_unit"] == pd.Timestamp("2026-08-31")
