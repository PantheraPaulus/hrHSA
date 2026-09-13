"""Timezone validation helpers shared across HSA workflows."""

from __future__ import annotations

import pandas as pd


def require_timezone_aware(
    values,
    *,
    name: str = "Timestamp",
    to_utc: bool = False,
):
    """Parse timestamps while refusing to guess a timezone.

    Timezone-naive timestamps are scientifically ambiguous for telemetry data.
    Callers must localize timestamps explicitly before entering hrHSA.  The
    original timezone is preserved by default; ``to_utc=True`` is reserved for
    boundaries such as ERA5/xarray lookup where UTC coordinates are required.
    """
    try:
        parsed = pd.to_datetime(values, errors="raise", utc=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r} contains timestamps that could not be parsed.") from exc

    if isinstance(parsed, pd.Series):
        timezone = parsed.dt.tz
    elif isinstance(parsed, pd.DatetimeIndex):
        timezone = parsed.tz
    elif isinstance(parsed, pd.Timestamp):
        timezone = parsed.tzinfo
    else:
        timezone = None

    if timezone is None:
        raise ValueError(
            f"{name!r} must contain timezone-aware timestamps. "
            "Timezone-naive timestamps are ambiguous and hrHSA will not assume UTC. "
            "Localize them explicitly before analysis, for example with "
            "pd.to_datetime(...).dt.tz_localize('Area/City')."
        )

    if not to_utc:
        return parsed

    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_convert("UTC")
    return parsed.tz_convert("UTC")


__all__ = ["require_timezone_aware"]
