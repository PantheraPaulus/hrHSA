"""Quick-look diagnostics for time-varying environmental fields.

These helpers are designed for questions such as "how different were thermal
conditions at 09:00 and 12:00 on this day?" without first sampling the field at
SSF candidate endpoints. They subset a bounded spatial domain, extract one or
more time slices, optionally transform/derive variables, summarize their
spatial distributions, and produce directly comparable maps/distributions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm, SymLogNorm
from hsa.ssf.environment import (
    _compute_with_retry,
    _match_longitude_convention,
    _normalize_dynamic_variables,
    sample_dynamic_covariates_at_points,
)


def _as_dataset(
    field: xr.Dataset | xr.DataArray,
    source_variables: Sequence[str],
) -> xr.Dataset:
    if isinstance(field, xr.Dataset):
        return field
    if not isinstance(field, xr.DataArray):
        raise TypeError("field must be an xarray Dataset or DataArray.")

    if field.name is None:
        if len(source_variables) != 1:
            raise ValueError(
                "An unnamed DataArray can only be used when one source variable "
                "is requested."
            )
        return field.to_dataset(name=source_variables[0])
    return field.to_dataset()


def _normalize_requested_times(
    times,
    *,
    timezone: str | None,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[str]]:
    if isinstance(times, (str, pd.Timestamp, np.datetime64)):
        values = [times]
    else:
        values = list(times)
    if not values:
        raise ValueError("At least one comparison time is required.")

    local_times: list[pd.Timestamp] = []
    utc_times: list[pd.Timestamp] = []
    labels: list[str] = []
    for value in values:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError(f"Invalid comparison time: {value!r}")

        if ts.tzinfo is None:
            if timezone is None:
                raise ValueError(
                    "Naive comparison times require an explicit timezone. "
                    "Use timezone='UTC' for ERA5 UTC clock times."
                )
            ts = ts.tz_localize(timezone)
        elif timezone is not None:
            ts = ts.tz_convert(timezone)

        utc = ts.tz_convert("UTC").tz_localize(None)
        local_times.append(ts)
        utc_times.append(utc)
        labels.append(ts.strftime("%Y-%m-%d %H:%M %Z"))

    return local_times, utc_times, labels


def _coord_slice(coord: xr.DataArray, lower: float, upper: float) -> slice:
    descending = float(coord.isel({coord.dims[0]: 0})) > float(
        coord.isel({coord.dims[0]: -1})
    )
    return slice(upper, lower) if descending else slice(lower, upper)


def _to_longitude_180(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    normalized = ((values + 180.0) % 360.0) - 180.0
    # Keep +180 rather than turning an explicit eastern edge into -180.
    normalized[(normalized == -180.0) & (values > 0)] = 180.0
    return normalized


def _select_longitude_window(
    ds: xr.Dataset,
    *,
    longitude_name: str,
    west: float,
    east: float,
) -> xr.Dataset:
    """Select a conventional -180..180 geographic window from any lon convention.

    ERA5 stores may expose longitude as 0..360. A perfectly ordinary study
    region that crosses Greenwich (for example -8..4 degrees) then converts to
    352..4, which superficially looks like a dateline crossing. This helper
    handles that storage seam by selecting both pieces and remapping only the
    already-subset coordinates back to -180..180 for plotting.
    """
    west, east = map(float, _to_longitude_180([west, east]))
    if west > east:
        raise ValueError(
            "True dateline-crossing quick-look domains are not yet supported. "
            "Use a domain whose western/eastern bounds are continuous in the "
            "-180..180 longitude convention."
        )

    lon = ds[longitude_name]
    lon_min = float(lon.min())
    lon_max = float(lon.max())
    converted_west, converted_east = map(
        float,
        _match_longitude_convention(np.asarray([west, east]), lon),
    )

    if converted_west <= converted_east:

        selected = ds.sel(
            {
                longitude_name:
                    _coord_slice(
                        lon,
                        converted_west,
                        converted_east,
                    )
            }
        )

        # Always return conventional -180..180 longitudes
        # for plotting, even if ERA5 stores them as 0..360.
        remapped = _to_longitude_180(
            selected[
                longitude_name
            ].values
        )

        selected = selected.assign_coords(
            {
                longitude_name:
                    remapped
            }
        )

        return selected.sortby(
            longitude_name
        )

    # Storage seam crossing (normally a 0..360 field with a Greenwich-crossing
    # geographic domain). Select the two small pieces separately so a remote
    # global store is never globally sorted/reindexed.
    first = ds.sel(
        {longitude_name: _coord_slice(lon, converted_west, lon_max)}
    )
    second = ds.sel(
        {longitude_name: _coord_slice(lon, lon_min, converted_east)}
    )
    pieces = []
    for piece in (first, second):
        if piece.sizes.get(longitude_name, 0) == 0:
            continue
        remapped = _to_longitude_180(piece[longitude_name].values)
        piece = piece.assign_coords({longitude_name: remapped})
        pieces.append(piece)
    if not pieces:
        raise ValueError("Requested longitude bounds do not overlap the dynamic field.")

    combined = xr.concat(pieces, dim=longitude_name)
    return combined.sortby(longitude_name)


def _native_spacing(coord: xr.DataArray) -> float:
    values = np.asarray(coord.values, dtype=float).ravel()
    if values.size < 2:
        return np.nan
    differences = np.abs(np.diff(values))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if not differences.size:
        return np.nan
    return float(np.median(differences))


def _decimate_spatial_grid(
    ds: xr.Dataset,
    *,
    longitude_name: str,
    latitude_name: str,
    resolution: float | None,
    max_cells: int | None,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Cheaply decimate a quick-look grid without averaging source cells.

    ``resolution`` is an approximate target spacing in geographic degrees.
    Decimation uses index strides so remote/Dask-backed fields do not need every
    native-resolution cell merely to make an exploratory plot. It is therefore
    intentionally a quick-look operation, not conservative spatial aggregation.
    """
    if resolution is not None:
        resolution = float(resolution)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be a positive number of degrees.")
    if max_cells is not None:
        max_cells = int(max_cells)
        if max_cells <= 0:
            raise ValueError("max_cells must be positive.")

    lon_stride = 1
    lat_stride = 1
    if resolution is not None:
        lon_spacing = _native_spacing(ds[longitude_name])
        lat_spacing = _native_spacing(ds[latitude_name])
        if np.isfinite(lon_spacing) and lon_spacing > 0:
            lon_stride = max(1, int(np.ceil(resolution / lon_spacing)))
        if np.isfinite(lat_spacing) and lat_spacing > 0:
            lat_stride = max(1, int(np.ceil(resolution / lat_spacing)))

    decimated = ds.isel(
        {
            longitude_name: slice(None, None, lon_stride),
            latitude_name: slice(None, None, lat_stride),
        }
    )

    auto_stride = 1
    n_cells = (
        decimated.sizes.get(longitude_name, 1)
        * decimated.sizes.get(latitude_name, 1)
    )
    if max_cells is not None and n_cells > max_cells:
        auto_stride = max(1, int(np.ceil(np.sqrt(n_cells / max_cells))))
        decimated = decimated.isel(
            {
                longitude_name: slice(None, None, auto_stride),
                latitude_name: slice(None, None, auto_stride),
            }
        )
        lon_stride *= auto_stride
        lat_stride *= auto_stride

    return decimated, {
        "longitude_stride": int(lon_stride),
        "latitude_stride": int(lat_stride),
        "resolution_requested_degrees": resolution,
        "max_cells_requested": max_cells,
        "n_spatial_cells": int(
            decimated.sizes.get(longitude_name, 1)
            * decimated.sizes.get(latitude_name, 1)
        ),
    }


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: Sequence[float],
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[keep]
    weights = weights[keep]
    if values.size == 0:
        return np.full(len(probabilities), np.nan, dtype=float)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative = (cumulative - 0.5 * weights) / cumulative[-1]
    return np.interp(np.asarray(probabilities, dtype=float), cumulative, values)


def _spatial_values_and_weights(
    data: xr.DataArray,
    *,
    latitude_name: str,
    area_weighted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data.values, dtype=float)
    if latitude_name not in data.coords:
        weights = np.ones_like(values, dtype=float)
        return values.ravel(), weights.ravel()

    latitude = np.asarray(data[latitude_name].values, dtype=float)
    if area_weighted:
        lat_weights = np.clip(np.cos(np.deg2rad(latitude)), 0.0, None)
    else:
        lat_weights = np.ones_like(latitude, dtype=float)

    lat_da = xr.DataArray(
        lat_weights,
        dims=data[latitude_name].dims,
        coords={latitude_name: data[latitude_name]},
    )
    weights = np.asarray(lat_da.broadcast_like(data).values, dtype=float)
    return values.ravel(), weights.ravel()


def extract_dynamic_condition_snapshots(
    field: xr.Dataset | xr.DataArray,
    *,
    variables: Sequence[str] | Mapping[str, str] | str,
    times,
    bounds: tuple[float, float, float, float] | None = None,
    timezone: str | None = "UTC",
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    time_name: str = "valid_time",
    spatial_margin: float = 0.0,
    resolution: float | None = None,
    max_cells: int | None = None,
    time_method: str = "nearest",
    time_tolerance: str | pd.Timedelta | None = "90min",
    transforms: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    derived: Mapping[str, Callable[[xr.Dataset], xr.DataArray | np.ndarray]] | None = None,
    retries: int = 5,
    initial_delay: float = 1.0,
) -> xr.Dataset:
    """Extract comparable gridded environmental snapshots at requested times.

    Parameters
    ----------
    field
        Time-varying xarray field. The defaults match ERA5-style coordinates.
    variables
        Source variables to retain. A mapping uses ``source -> output`` naming,
        matching :func:`sample_dynamic_covariates_at_points`.
    times
        One or more timestamps. Naive timestamps are interpreted in ``timezone``
        and converted to UTC for lookup. This makes local-clock comparisons such
        as 09:00 versus 12:00 explicit while respecting ERA5's UTC time axis.
    bounds
        Geographic ``(west, south, east, north)`` bounds in conventional
        -180..180 longitude degrees. Ordinary regions crossing Greenwich are
        handled even when the backing field stores longitudes as 0..360.
    resolution
        Optional approximate target map spacing in degrees. The quick-look field
        is decimated by index stride before loading; no spatial averaging is
        performed. For example ``resolution=0.5`` makes a much cheaper overview
        from a 0.1-degree source field.
    max_cells
        Optional upper target for latitude x longitude cells after subsetting.
        If necessary, an additional equal stride is applied in both dimensions.
    transforms
        Per-source transformations applied to sampled arrays, e.g.
        ``{"sshf": lambda x: -x / 3600}``.
    derived
        Optional derived variables computed from each already-loaded time slice.
        Each callable receives the selected source-variable Dataset and should
        return a spatial DataArray or ndarray.
    time_method
        ``"nearest"`` or ``"linear"`` temporal selection/interpolation.

    Returns
    -------
    xarray.Dataset
        Dataset with a ``comparison_time`` dimension and coordinates containing
        both requested display labels and matched UTC timestamps.
    """
    if time_method not in {"nearest", "linear"}:
        raise ValueError("time_method must be 'nearest' or 'linear'.")
    if spatial_margin < 0:
        raise ValueError("spatial_margin must be non-negative.")
    if retries <= 0:
        raise ValueError("retries must be positive.")

    variable_map = _normalize_dynamic_variables(variables)
    source_variables = list(variable_map)
    transforms = {} if transforms is None else dict(transforms)
    derived = {} if derived is None else dict(derived)
    if not derived and not source_variables:
        raise ValueError("At least one source or derived variable is required.")

    unknown_transforms = sorted(set(transforms).difference(source_variables))
    if unknown_transforms:
        raise KeyError(
            "Transforms were supplied for variables that are not requested: "
            f"{unknown_transforms}"
        )
    duplicate_outputs = set(variable_map.values()).intersection(derived)
    if duplicate_outputs:
        raise ValueError(
            "Derived variable names collide with sampled output names: "
            f"{sorted(duplicate_outputs)}"
        )

    ds = _as_dataset(field, source_variables)
    for name in (longitude_name, latitude_name, time_name):
        if name not in ds.coords:
            raise KeyError(f"{name!r} not found in dynamic-field coordinates.")
    missing = sorted(set(source_variables).difference(ds.data_vars))
    if missing:
        raise KeyError(f"Dynamic variables not found: {missing}")

    _, utc_times, labels = _normalize_requested_times(times, timezone=timezone)

    lon = ds[longitude_name]
    lat = ds[latitude_name]
    lat_min = float(lat.min())
    lat_max = float(lat.max())

    if bounds is None:
        field_longitudes = _to_longitude_180(lon.values)
        west = float(np.nanmin(field_longitudes))
        east = float(np.nanmax(field_longitudes))
        south, north = lat_min, lat_max
    else:
        if len(bounds) != 4:
            raise ValueError("bounds must be (west, south, east, north).")
        west_raw, south, east_raw, north = map(float, bounds)
        if south >= north:
            raise ValueError("bounds must satisfy south < north.")
        west, east = map(float, _to_longitude_180([west_raw, east_raw]))
        if west > east:
            raise ValueError(
                "True dateline-crossing quick-look domains are not yet supported."
            )

    west = max(-180.0, west - spatial_margin)
    east = min(180.0, east + spatial_margin)
    south = max(lat_min, south - spatial_margin)
    north = min(lat_max, north + spatial_margin)
    if west >= east or south >= north:
        raise ValueError("Requested bounds do not overlap the dynamic field.")

    spatial = ds[source_variables].sel(
        {latitude_name: _coord_slice(lat, south, north)}
    )
    spatial = _select_longitude_window(
        spatial,
        longitude_name=longitude_name,
        west=west,
        east=east,
    )
    if (
        spatial.sizes.get(longitude_name, 0) == 0
        or spatial.sizes.get(latitude_name, 0) == 0
    ):
        raise ValueError("Requested bounds do not overlap the dynamic field.")

    spatial, decimation = _decimate_spatial_grid(
        spatial,
        longitude_name=longitude_name,
        latitude_name=latitude_name,
        resolution=resolution,
        max_cells=max_cells,
    )

    pieces: list[xr.Dataset] = []
    tolerance = None if time_tolerance is None else pd.Timedelta(time_tolerance)
    for utc, label in zip(utc_times, labels, strict=True):
        target = np.datetime64(utc.to_datetime64())
        if time_method == "nearest":
            kwargs: dict[str, Any] = {"method": "nearest"}
            if tolerance is not None:
                kwargs["tolerance"] = tolerance
            selected = spatial.sel({time_name: target}, **kwargs)
        else:
            selected = spatial.interp({time_name: target})

        selected = _compute_with_retry(
            selected,
            retries=retries,
            initial_delay=initial_delay,
        )
        matched = (
            pd.Timestamp(selected[time_name].values)
            if time_name in selected.coords
            else utc
        )

        output = xr.Dataset()
        for source, output_name in variable_map.items():
            source_da = selected[source]
            values = np.asarray(source_da.values)
            if source in transforms:
                values = np.asarray(transforms[source](values))
            if values.shape != source_da.shape:
                raise ValueError(
                    f"Transform for {source!r} changed shape from "
                    f"{source_da.shape} to {values.shape}."
                )
            output[output_name] = xr.DataArray(
                values,
                dims=source_da.dims,
                coords=source_da.coords,
                attrs=dict(source_da.attrs),
            )

        for output_name, function in derived.items():
            value = function(selected)
            if isinstance(value, xr.DataArray):
                derived_da = value
            else:
                values = np.asarray(value)
                spatial_dims = tuple(
                    dim for dim in selected[source_variables[0]].dims if dim != time_name
                )
                reference = selected[source_variables[0]]
                if values.shape != reference.shape:
                    raise ValueError(
                        f"Derived variable {output_name!r} has shape {values.shape}; "
                        f"expected {reference.shape}."
                    )
                derived_da = xr.DataArray(
                    values,
                    dims=spatial_dims,
                    coords={dim: reference.coords[dim] for dim in spatial_dims},
                )
            output[output_name] = derived_da

        if time_name in output.coords:
            output = output.drop_vars(time_name)
        output = output.expand_dims(comparison_time=[target])
        output = output.assign_coords(
            requested_time_label=("comparison_time", [label]),
            matched_time_utc=("comparison_time", [np.datetime64(matched.to_datetime64())]),
        )
        pieces.append(output)

    result = xr.concat(pieces, dim="comparison_time")
    result.attrs.update(
        {
            "comparison_timezone": "mixed/aware" if timezone is None else str(timezone),
            "bounds_west_south_east_north": (west, south, east, north),
            "time_method": time_method,
            **decimation,
        }
    )
    return result


def summarize_dynamic_conditions(
    snapshots: xr.Dataset,
    *,
    latitude_name: str = "latitude",
    area_weighted: bool = True,
) -> pd.DataFrame:
    """Summarize each variable/time slice over the mapped spatial domain."""
    if "comparison_time" not in snapshots.dims:
        raise ValueError("snapshots must contain a 'comparison_time' dimension.")

    rows: list[dict[str, Any]] = []
    probabilities = [0.05, 0.25, 0.50, 0.75, 0.95]
    labels = snapshots.coords.get("requested_time_label")

    for variable in snapshots.data_vars:
        for i in range(snapshots.sizes["comparison_time"]):
            data = snapshots[variable].isel(comparison_time=i)
            values, weights = _spatial_values_and_weights(
                data,
                latitude_name=latitude_name,
                area_weighted=area_weighted,
            )
            keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            v = values[keep]
            w = weights[keep]
            if v.size:
                mean = float(np.average(v, weights=w))
                sd = float(np.sqrt(np.average((v - mean) ** 2, weights=w)))
                q05, q25, q50, q75, q95 = _weighted_quantile(v, w, probabilities)
                positive = float(np.average(v > 0, weights=w))
            else:
                mean = sd = q05 = q25 = q50 = q75 = q95 = positive = np.nan

            timestamp = pd.Timestamp(snapshots["comparison_time"].values[i])
            label = str(labels.values[i]) if labels is not None else str(timestamp)
            rows.append(
                {
                    "variable": variable,
                    "time": timestamp,
                    "time_label": label,
                    "n_cells": int(v.size),
                    "mean": mean,
                    "sd": sd,
                    "q05": float(q05),
                    "q25": float(q25),
                    "median": float(q50),
                    "q75": float(q75),
                    "q95": float(q95),
                    "fraction_positive": positive,
                    "area_weighted": bool(area_weighted),
                }
            )
    return pd.DataFrame(rows)

def _make_publication_norm(
    values,
    *,
    mode="symlog",
    robust=True,
    quantile_range=(0.01, 0.99),
    center=0.0,
    linthresh=None,
):
    """
    Construct a shared normalization for publication plots.

    mode:
        "linear"
        "centered"
        "symlog"
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return Normalize(0, 1), (0, 1)

    if robust:
        qlo, qhi = quantile_range
        vmin, vmax = np.quantile(
            values,
            [qlo, qhi],
        )
    else:
        vmin = values.min()
        vmax = values.max()

    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    if mode == "linear":

        norm = Normalize(
            vmin=vmin,
            vmax=vmax,
        )

    elif mode == "centered":

        # Make the positive and negative sides use
        # comparable visual space around zero.
        span = max(
            abs(vmin - center),
            abs(vmax - center),
        )

        vmin = center - span
        vmax = center + span

        norm = TwoSlopeNorm(
            vmin=vmin,
            vcenter=center,
            vmax=vmax,
        )

    elif mode == "symlog":

        span = max(
            abs(vmin - center),
            abs(vmax - center),
        )

        # A reasonable automatic linear range around zero.
        if linthresh is None:
            linthresh = max(
                span * 0.05,
                np.finfo(float).eps,
            )

        norm = SymLogNorm(
            linthresh=linthresh,
            linscale=1.0,
            vmin=vmin,
            vmax=vmax,
            base=10,
        )

    elif mode == "diverging":

        if not vmin < center < vmax:
            norm = Normalize(
                vmin=vmin,
                vmax=vmax,
            )
        else:
            norm = TwoSlopeNorm(
                vmin=vmin,
                vcenter=center,
                vmax=vmax,
            )

    else:
        raise ValueError(
            "mode must be 'linear', "
            "'centered', 'symlog', or 'diverging'."
        )

    return norm, (vmin, vmax)

def plot_dynamic_condition_comparison(
    snapshots: xr.Dataset,
    *,
    kind: str = "both",
    distribution_kind: str = "hist",
    bins: int = 40,
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    area_weighted: bool = True,
    variable_labels: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    cmaps: Mapping[str, str] | None = None,
    robust: bool = True,
    quantile_range: tuple[float, float] = (0.01, 0.99),
    figsize: tuple[float, float] | None = None,
    norm_mode: str = "symlog",
    norm_center: float = 0.0,
    linthresh: float | None = None,
    map_height: float = 1.0,
    distribution_height: float = 1.0,
    colorbar_height: float = 0.055,
    show_time_labels: bool = True
):
    """
    Publication-style comparison of dynamic environmental conditions.

    With kind="both", time-specific maps are arranged horizontally
    across the upper part of each variable block, with the spatial
    distribution spanning the full width underneath.

    Example for five times:

        map  map  map  map  map
        -------- colorbar -------
        spatial distribution
    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    if kind not in {"both", "distribution", "map"}:
        raise ValueError(
            "kind must be 'both', 'distribution', or 'map'."
        )

    if distribution_kind not in {"hist", "ecdf"}:
        raise ValueError(
            "distribution_kind must be 'hist' or 'ecdf'."
        )

    if bins <= 1:
        raise ValueError(
            "bins must be greater than 1."
        )

    qlo, qhi = map(
        float,
        quantile_range,
    )

    if not 0 <= qlo < qhi <= 1:
        raise ValueError(
            "quantile_range must satisfy "
            "0 <= low < high <= 1."
        )

    variables = list(
        snapshots.data_vars
    )

    if not variables:
        raise ValueError(
            "snapshots contains no variables to plot."
        )

    n_times = snapshots.sizes.get(
        "comparison_time",
        0,
    )

    if n_times == 0:
        raise ValueError(
            "snapshots contains no comparison times."
        )

    variable_labels = (
        {}
        if variable_labels is None
        else dict(variable_labels)
    )

    units = (
        {}
        if units is None
        else dict(units)
    )

    cmaps = (
        {}
        if cmaps is None
        else dict(cmaps)
    )

    # ------------------------------------------------------------
    # Time labels
    # ------------------------------------------------------------

    labels_coord = snapshots.coords.get(
        "requested_time_label"
    )

    if labels_coord is not None:
        raw_time_labels = [
            str(v)
            for v in labels_coord.values
        ]
    else:
        raw_time_labels = [
            str(pd.Timestamp(v))
            for v in snapshots[
                "comparison_time"
            ].values
        ]

    time_labels = []

    for raw in raw_time_labels:

        try:

            ts = pd.Timestamp(raw)

            if pd.isna(ts):
                raise ValueError

            time_labels.append(
                ts.strftime("%H:%M")
            )

        except Exception:

            parts = raw.split()

            clock = next(
                (
                    p
                    for p in parts
                    if ":" in p
                ),
                raw,
            )

            time_labels.append(
                clock
            )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    n_variables = len(
        variables
    )

    if kind == "both":

        if figsize is None:

            # Width increases somewhat with number of time slices,
            # but is capped so large time series remain reasonable.
            width = max(
                10.0,
                min(
                    3.0 * n_times,
                    16.0,
                ),
            )

            height = (
                5.8 * n_variables
            )

            figsize = (
                width,
                height,
            )

    elif kind == "map":

        if figsize is None:

            figsize = (
                max(
                    8.0,
                    min(
                        3.0 * n_times,
                        16.0,
                    ),
                ),
                3.1 * n_variables,
            )

    else:

        if figsize is None:

            figsize = (
                8.5,
                3.8 * n_variables,
            )

    fig = plt.figure(
        figsize=figsize,
        constrained_layout=True,
    )

    outer = fig.add_gridspec(
        nrows=n_variables,
        ncols=1,
        hspace=0.18,
    )

    axes = {
        "maps": {},
        "distributions": {},
        "colorbars": {},
    }

    # ------------------------------------------------------------
    # Each variable
    # ------------------------------------------------------------

    for variable_idx, variable in enumerate(
        variables
    ):

        values_all = np.asarray(
            snapshots[variable].values,
            dtype=float,
        ).ravel()

        values_all = values_all[
            np.isfinite(values_all)
        ]

        # --------------------------------------------------------
        # Shared normalization
        # --------------------------------------------------------

        norm, limits = _make_publication_norm(
            values_all,
            mode=norm_mode,
            robust=robust,
            quantile_range=quantile_range,
            center=norm_center,
            linthresh=linthresh,
        )

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------

        pretty = variable_labels.get(
            variable,
            variable
            .replace("_", " ")
            .title(),
        )

        unit = units.get(
            variable,
            snapshots[
                variable
            ].attrs.get(
                "units",
                "",
            ),
        )

        axis_label = (
            f"{pretty} [{unit}]"
            if unit
            else pretty
        )

        cmap = cmaps.get(
            variable,
            None,
        )

        # --------------------------------------------------------
        # Variable layout
        # --------------------------------------------------------

        if kind == "both":

            # 3 rows:
            #
            #  maps
            #  horizontal colorbar
            #  distribution
            #
            block = outer[
                variable_idx
            ].subgridspec(
                nrows=3,
                ncols=n_times,
                height_ratios=[
                    map_height,
                    colorbar_height,
                    distribution_height,
                ],
                hspace=0.12,
                wspace=0.06,
            )

            map_axes = [
                fig.add_subplot(
                    block[0, i]
                )
                for i in range(n_times)
            ]

            cax = fig.add_subplot(
                block[1, :]
            )

            dist_ax = fig.add_subplot(
                block[2, :]
            )

        elif kind == "map":

            block = outer[
                variable_idx
            ].subgridspec(
                nrows=2,
                ncols=n_times,
                height_ratios=[
                    1.0,
                    0.055,
                ],
                hspace=0.10,
                wspace=0.06,
            )

            map_axes = [
                fig.add_subplot(
                    block[0, i]
                )
                for i in range(n_times)
            ]

            cax = fig.add_subplot(
                block[1, :]
            )

            dist_ax = None

        else:

            block = outer[
                variable_idx
            ].subgridspec(
                nrows=1,
                ncols=1,
            )

            map_axes = []

            cax = None

            dist_ax = fig.add_subplot(
                block[0, 0]
            )

        axes["maps"][
            variable
        ] = map_axes

        axes["distributions"][
            variable
        ] = dist_ax

        axes["colorbars"][
            variable
        ] = cax

        # --------------------------------------------------------
        # Maps
        # --------------------------------------------------------

        mappable = None

        for time_idx, ax in enumerate(
            map_axes
        ):

            data = snapshots[
                variable
            ].isel(
                comparison_time=time_idx
            )

            kwargs = {
                "ax": ax,
                "x": longitude_name,
                "y": latitude_name,
                "add_colorbar": False,
                "norm": norm,
            }

            if cmap is not None:
                kwargs[
                    "cmap"
                ] = cmap

            mappable = (
                data.plot.pcolormesh(
                    **kwargs
                )
            )

            # --------------------------------------------
            # Time label
            # --------------------------------------------

            if show_time_labels:
                ax.text(
                    0.03,
                    0.96,
                    time_labels[time_idx],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9.5,
                    fontweight="semibold",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 2.2,
                    },
                )

            ax.set_title("")

            # Only first map needs y-axis label.
            if time_idx == 0:
                ax.set_ylabel(
                    "Latitude"
                )
            else:
                ax.set_ylabel("")
                ax.tick_params(
                    axis="y",
                    labelleft=False,
                )

            ax.set_xlabel(
                "Longitude"
            )

            ax.xaxis.set_major_locator(
                MaxNLocator(
                    nbins=4,
                )
            )

            ax.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=4,
                )
            )

            ax.tick_params(
                labelsize=8.5,
                length=3,
            )

            for spine in (
                ax.spines.values()
            ):
                spine.set_linewidth(
                    0.6
                )

            # Variable title over first panel
            if time_idx == 0:

                ax.set_title(
                    pretty,
                    loc="left",
                    fontsize=12,
                    fontweight="bold",
                    pad=7,
                )

        # --------------------------------------------------------
        # Shared horizontal colorbar
        # --------------------------------------------------------

        if (
            mappable is not None
            and cax is not None
        ):

            cbar = fig.colorbar(
                mappable,
                cax=cax,
                orientation="horizontal",
            )

            cbar.set_label(
                axis_label,
                fontsize=9,
                labelpad=4,
            )

            cbar.ax.tick_params(
                labelsize=8,
                length=2.5,
            )

            cbar.outline.set_linewidth(
                0.5
            )

        # --------------------------------------------------------
        # Distribution
        # --------------------------------------------------------

        if dist_ax is not None:

            if distribution_kind == "hist":

                edges = np.linspace(
                    limits[0],
                    limits[1],
                    bins,
                )

                for time_idx in range(
                    n_times
                ):

                    data = snapshots[
                        variable
                    ].isel(
                        comparison_time=time_idx
                    )

                    values, weights = (
                        _spatial_values_and_weights(
                            data,
                            latitude_name=
                                latitude_name,
                            area_weighted=
                                area_weighted,
                        )
                    )

                    keep = (
                        np.isfinite(values)
                        & np.isfinite(weights)
                        & (weights > 0)
                        & (
                            values
                            >= limits[0]
                        )
                        & (
                            values
                            <= limits[1]
                        )
                    )

                    dist_ax.hist(
                        values[keep],
                        bins=edges,
                        weights=
                            weights[keep],
                        density=True,
                        histtype="step",
                        linewidth=1.8,
                        label=
                            time_labels[
                                time_idx
                            ],
                    )

                dist_ax.set_ylabel(
                    (
                        "Area-weighted density"
                        if area_weighted
                        else "Density"
                    )
                )

            else:

                for time_idx in range(
                    n_times
                ):

                    data = snapshots[
                        variable
                    ].isel(
                        comparison_time=time_idx
                    )

                    values, weights = (
                        _spatial_values_and_weights(
                            data,
                            latitude_name=
                                latitude_name,
                            area_weighted=
                                area_weighted,
                        )
                    )

                    keep = (
                        np.isfinite(values)
                        & np.isfinite(weights)
                        & (weights > 0)
                    )

                    values = values[
                        keep
                    ]

                    weights = weights[
                        keep
                    ]

                    order = np.argsort(
                        values
                    )

                    values = values[
                        order
                    ]

                    weights = weights[
                        order
                    ]

                    if values.size:

                        cdf = (
                            np.cumsum(
                                weights
                            )
                            / weights.sum()
                        )

                        dist_ax.step(
                            values,
                            cdf,
                            where="post",
                            linewidth=1.8,
                            label=
                                time_labels[
                                    time_idx
                                ],
                        )

                dist_ax.set_ylabel(
                    (
                        "Area-weighted ECDF"
                        if area_weighted
                        else "ECDF"
                    )
                )

                dist_ax.set_ylim(
                    0,
                    1,
                )

            # ----------------------------------------------------
            # Distribution formatting
            # ----------------------------------------------------

            dist_ax.set_xlim(
                *limits
            )

            dist_ax.set_xlabel(
                axis_label
            )

            dist_ax.set_title(
                "Spatial distribution",
                loc="left",
                fontsize=11,
                fontweight="bold",
                pad=7,
            )

            if (
                limits[0]
                < 0
                < limits[1]
            ):

                dist_ax.axvline(
                    0,
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.55,
                )

            dist_ax.grid(
                axis="y",
                linewidth=0.5,
                alpha=0.22,
            )

            dist_ax.tick_params(
                labelsize=9,
                length=3,
            )

            dist_ax.xaxis.set_major_locator(
                MaxNLocator(
                    nbins=8,
                )
            )

            dist_ax.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=5,
                )
            )

            dist_ax.spines[
                "top"
            ].set_visible(
                False
            )

            dist_ax.spines[
                "right"
            ].set_visible(
                False
            )

            dist_ax.legend(
                title="Time",
                frameon=False,
                fontsize=9,
                title_fontsize=9,
                ncol=min(
                    n_times,
                    5,
                ),
                loc="upper right",
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = summarize_dynamic_conditions(
        snapshots,
        latitude_name=latitude_name,
        area_weighted=area_weighted,
    )

    return {
        "figure": fig,
        "axes": axes,
        "summary": summary,
        "snapshots": snapshots,
    }

def plot_static_conditions(
    field: xr.Dataset | xr.DataArray,
    *,
    bands,
    band_dim: str = "band",
    x_name: str = "x",
    y_name: str = "y",
    domain: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    max_cells: int | None = None,
    variable_labels: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    cmaps: Mapping[str, str] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
):
    """
    Inspect one or more static environmental raster bands using the same
    publication-style map/distribution layout as dynamic conditions.

    Parameters
    ----------
    field
        Raster DataArray or Dataset.

        Typical hrHSA static raster:
            DataArray(band, y, x)

    bands
        Band name or sequence of band names.

    domain
        Optional spatial bounds:
            (xmin, ymin, xmax, ymax)

        in the native CRS of the raster.

    resolution
        Optional approximate target spatial resolution in native CRS units.
        For a projected raster this will usually be metres.

    max_cells
        Optional approximate maximum number of raster cells retained for
        the quick-look diagnostic.
    """

    # ------------------------------------------------------------
    # Normalize bands
    # ------------------------------------------------------------

    if isinstance(bands, str):
        bands = [bands]
    else:
        bands = list(bands)

    if not bands:
        raise ValueError(
            "At least one band must be requested."
        )

    # ------------------------------------------------------------
    # Convert Dataset -> DataArray where possible
    # ------------------------------------------------------------

    if isinstance(field, xr.Dataset):

        # Case 1:
        # Dataset contains a single DataArray with a band dimension.
        candidates = [
            name
            for name, da in field.data_vars.items()
            if band_dim in da.dims
        ]

        if len(candidates) == 1:

            raster = field[
                candidates[0]
            ]

        # Case 2:
        # requested bands are individual Dataset variables.
        elif all(
            band in field.data_vars
            for band in bands
        ):

            arrays = [
                field[band]
                .expand_dims(
                    {band_dim: [band]}
                )
                for band in bands
            ]

            raster = xr.concat(
                arrays,
                dim=band_dim,
            )

        else:

            raise ValueError(
                "Could not identify the static raster. "
                "Expected either one DataArray with a "
                f"{band_dim!r} dimension or Dataset variables "
                "matching the requested bands."
            )

    elif isinstance(
        field,
        xr.DataArray,
    ):

        raster = field

    else:

        raise TypeError(
            "field must be an xarray Dataset or DataArray."
        )

    # ------------------------------------------------------------
    # Select requested bands
    # ------------------------------------------------------------

    if band_dim not in raster.dims:

        if len(bands) != 1:

            raise ValueError(
                f"Raster has no {band_dim!r} dimension, "
                "so only one band can be requested."
            )

        raster = raster.expand_dims(
            {
                band_dim:
                    [bands[0]]
            }
        )

    else:

        raster = raster.sel(
            {
                band_dim:
                    bands
            }
        )

    # ------------------------------------------------------------
    # Spatial subset
    # ------------------------------------------------------------

    if x_name not in raster.coords:
        raise KeyError(
            f"{x_name!r} not found in raster coordinates."
        )

    if y_name not in raster.coords:
        raise KeyError(
            f"{y_name!r} not found in raster coordinates."
        )

    if domain is not None:

        xmin, ymin, xmax, ymax = map(
            float,
            domain,
        )

        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                "domain must be "
                "(xmin, ymin, xmax, ymax)."
            )

        raster = raster.sel(
            {
                x_name:
                    _coord_slice(
                        raster[x_name],
                        xmin,
                        xmax,
                    ),
                y_name:
                    _coord_slice(
                        raster[y_name],
                        ymin,
                        ymax,
                    ),
            }
        )

    # ------------------------------------------------------------
    # Optional cheap decimation
    # ------------------------------------------------------------

    x_stride = 1
    y_stride = 1

    if resolution is not None:

        resolution = float(
            resolution
        )

        dx = _native_spacing(
            raster[x_name]
        )

        dy = _native_spacing(
            raster[y_name]
        )

        if (
            np.isfinite(dx)
            and dx > 0
        ):
            x_stride = max(
                1,
                int(
                    np.ceil(
                        resolution / dx
                    )
                ),
            )

        if (
            np.isfinite(dy)
            and dy > 0
        ):
            y_stride = max(
                1,
                int(
                    np.ceil(
                        resolution / dy
                    )
                ),
            )

    raster = raster.isel(
        {
            x_name:
                slice(
                    None,
                    None,
                    x_stride,
                ),
            y_name:
                slice(
                    None,
                    None,
                    y_stride,
                ),
        }
    )

    # Additional automatic decimation
    if max_cells is not None:

        max_cells = int(
            max_cells
        )

        n_cells = (
            raster.sizes[x_name]
            * raster.sizes[y_name]
        )

        if n_cells > max_cells:

            stride = max(
                1,
                int(
                    np.ceil(
                        np.sqrt(
                            n_cells
                            / max_cells
                        )
                    )
                ),
            )

            raster = raster.isel(
                {
                    x_name:
                        slice(
                            None,
                            None,
                            stride,
                        ),
                    y_name:
                        slice(
                            None,
                            None,
                            stride,
                        ),
                }
            )

    # ------------------------------------------------------------
    # Convert static bands into the same Dataset structure
    # expected by plot_dynamic_condition_comparison()
    # ------------------------------------------------------------

    variables = {}

    for band in bands:

        da = raster.sel(
            {
                band_dim:
                    band
            }
        )

        # Remove scalar band coordinate
        if band_dim in da.coords:
            da = da.drop_vars(
                band_dim
            )

        variables[band] = da

    snapshots = xr.Dataset(
        variables
    )

    # Add synthetic comparison dimension.
    snapshots = snapshots.expand_dims(
        comparison_time=[
            np.datetime64(
                "2000-01-01"
            )
        ]
    )

    snapshots = (
        snapshots.assign_coords(
            requested_time_label=(
                "comparison_time",
                [""],
            )
        )
    )

    # ------------------------------------------------------------
    # Plot using the exact same engine
    # ------------------------------------------------------------

    options = (
        {}
        if plot_kwargs is None
        else dict(plot_kwargs)
    )

    options.setdefault(
        "kind",
        "both",
    )

    options.setdefault(
        "show_time_labels",
        False,
    )

    options.setdefault(
        "latitude_name",
        y_name,
    )

    options.setdefault(
        "longitude_name",
        x_name,
    )

    # Latitude area-weighting makes no sense for a projected x/y raster.
    options.setdefault(
        "area_weighted",
        False,
    )

    if variable_labels is not None:
        options.setdefault(
            "variable_labels",
            variable_labels,
        )

    if units is not None:
        options.setdefault(
            "units",
            units,
        )

    if cmaps is not None:
        options.setdefault(
            "cmaps",
            cmaps,
        )

    return plot_dynamic_condition_comparison(
        snapshots,
        **options,
    )

def plot_stratum_environment(
    choices,
    field,
    *,
    stratum_id,
    variable,
    output_name=None,
    time_col="end_time",
    longitude_name="longitude",
    latitude_name="latitude",
    time_name="valid_time",
    transform=None,

    # spatial extent
    margin_fraction=0.25,
    minimum_margin_deg=0.05,
    minimum_context_cells: int = 4,

    # plotting
    cmap="RdBu_r",
    norm_mode="symlog",
    robust=True,
    quantile_range=(0.01, 0.99),
    linthresh=None,
    raster_alpha=0.55,

    distribution_kind="hist",
    bins: int | str = "fd",

    figsize=(11, 6),

    available_line_alpha=0.25,
    available_rug_alpha=0.65,

    variable_label=None,
    unit=None,
):
    """
    Publication-style environmental diagnostic for one SSF stratum.

    Left:
        Environmental raster at the stratum time with the observed
        and available candidate steps overlaid.

    Right:
        Distribution of the local environmental raster, plus the
        environmental values encountered by available endpoints and
        the observed endpoint.
    """

    import matplotlib.pyplot as plt
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    # ------------------------------------------------------------
    # Select stratum
    # ------------------------------------------------------------

    s = choices.loc[
        choices["stratum_id"] == stratum_id
    ].copy()

    if s.empty:
        raise ValueError(
            f"Stratum {stratum_id!r} not found."
        )

    if s["used"].sum() != 1:
        raise ValueError(
            "Stratum must contain exactly one used alternative."
        )

    available = s.loc[
        s["used"] == 0
    ].copy()

    observed = s.loc[
        s["used"] == 1
    ].copy()

    # ------------------------------------------------------------
    # Work in lon/lat because ERA5 is geographic
    # ------------------------------------------------------------

    endpoints_ll = s.to_crs(4326)

    start_geom = gpd.GeoSeries(
        s["start_geometry"],
        index=s.index,
        crs=s.crs,
    ).to_crs(4326)

    start_lon = start_geom.iloc[0].x
    start_lat = start_geom.iloc[0].y

    available_ll = endpoints_ll.loc[
        available.index
    ]

    observed_ll = endpoints_ll.loc[
        observed.index
    ]

    # ------------------------------------------------------------
    # Local spatial extent
    # ------------------------------------------------------------

    all_lon = np.r_[
        endpoints_ll.geometry.x.to_numpy(),
        start_lon,
    ]

    all_lat = np.r_[
        endpoints_ll.geometry.y.to_numpy(),
        start_lat,
    ]

    west = np.nanmin(all_lon)
    east = np.nanmax(all_lon)
    south = np.nanmin(all_lat)
    north = np.nanmax(all_lat)

    lon_range = max(
        east - west,
        minimum_margin_deg,
    )

    lat_range = max(
        north - south,
        minimum_margin_deg,
    )

    # Native atmospheric grid resolution
    lon_spacing = _native_spacing(
        field[longitude_name]
    )

    lat_spacing = _native_spacing(
        field[latitude_name]
    )

    # Ensure several ERA5 cells of context exist
    # around the complete candidate choice set.
    lon_grid_margin = (
        minimum_context_cells * lon_spacing
        if np.isfinite(lon_spacing)
        else 0.0
    )

    lat_grid_margin = (
        minimum_context_cells * lat_spacing
        if np.isfinite(lat_spacing)
        else 0.0
    )

    lon_margin = max(
        lon_range * margin_fraction,
        minimum_margin_deg,
        lon_grid_margin,
    )

    lat_margin = max(
        lat_range * margin_fraction,
        minimum_margin_deg,
        lat_grid_margin,
    )

    bounds = (
        west - lon_margin,
        south - lat_margin,
        east + lon_margin,
        north + lat_margin,
    )

    # ------------------------------------------------------------
    # Stratum time
    # ------------------------------------------------------------

    timestamp = pd.Timestamp(
        s[time_col].iloc[0]
    )

    # ------------------------------------------------------------
    # Extract environmental raster
    #
    # Uses the quick-look extractor we already wrote.
    # ------------------------------------------------------------

    transforms = (
        {variable: transform}
        if transform is not None
        else None
    )

    output_name = (
        variable
        if output_name is None
        else output_name
    )

    variable_mapping = {
        variable: output_name,
    }

    snapshots = extract_dynamic_condition_snapshots(
        field,
        variables=variable_mapping,
        times=[timestamp],
        bounds=bounds,
        timezone=None,
        longitude_name=longitude_name,
        latitude_name=latitude_name,
        time_name=time_name,
        transforms=transforms,
        time_method="nearest",
    )

    raster = snapshots[
        output_name
    ].isel(
        comparison_time=0
    )

    # ------------------------------------------------------------
    # Sample the same variable at candidate endpoints
    # ------------------------------------------------------------

    candidate_points = sample_dynamic_covariates_at_points(
        s,
        field,
        variables=variable_mapping,
        time_col=time_col,
        longitude_name=longitude_name,
        latitude_name=latitude_name,
        time_name=time_name,
        method="linear",
        transforms=transforms,
    )

    available_values = (
        candidate_points.loc[
            candidate_points["used"] == 0,
            output_name,
        ]
        .to_numpy(dtype=float)
    )

    observed_value = float(
        candidate_points.loc[
            candidate_points["used"] == 1,
            output_name,
        ].iloc[0]
    )

    # ------------------------------------------------------------
    # Shared normalization
    # ------------------------------------------------------------

    raster_values = np.asarray(
        raster.values,
        dtype=float,
    )

    norm, limits = _make_publication_norm(
        raster_values,
        mode=norm_mode,
        robust=robust,
        quantile_range=quantile_range,
        center=0.0,
        linthresh=linthresh,
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=figsize,
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[
            1.35,
            0.045,
            1.0,
        ],
        wspace=0.10,
    )

    ax_map = fig.add_subplot(
        gs[0, 0]
    )

    cax = fig.add_subplot(
        gs[0, 1]
    )

    ax_dist = fig.add_subplot(
        gs[0, 2]
    )

    # ------------------------------------------------------------
    # Raster
    # ------------------------------------------------------------

    mappable = raster.plot.pcolormesh(
        ax=ax_map,
        x=longitude_name,
        y=latitude_name,
        cmap=cmap,
        norm=norm,
        alpha=raster_alpha,
        add_colorbar=False,
    )

    ax_map.set_title(
        "",
        loc="center",
    )

    ax_map.set_title(
        "",
        loc="right",
    )

    # ------------------------------------------------------------
    # Available candidate steps
    # ------------------------------------------------------------

    for geometry in available_ll.geometry:

        ax_map.plot(
            [
                start_lon,
                geometry.x,
            ],
            [
                start_lat,
                geometry.y,
            ],
            linewidth=0.8,
            alpha=available_line_alpha,
            color="0.25",
            zorder=4,
        )

    # ------------------------------------------------------------
    # Observed step
    # ------------------------------------------------------------

    observed_geometry = (
        observed_ll.geometry.iloc[0]
    )

    ax_map.plot(
        [
            start_lon,
            observed_geometry.x,
        ],
        [
            start_lat,
            observed_geometry.y,
        ],
        linewidth=2.8,
        color="black",
        label="Observed step",
        zorder=6,
    )

    # ------------------------------------------------------------
    # Step origin
    # ------------------------------------------------------------

    ax_map.scatter(
        start_lon,
        start_lat,
        marker="x",
        s=90,
        linewidth=2,
        color="black",
        label="Step origin",
        zorder=8,
    )

    # ------------------------------------------------------------
    # Available endpoints
    # ------------------------------------------------------------

    ax_map.scatter(
        available_ll.geometry.x,
        available_ll.geometry.y,
        s=24,
        facecolor="white",
        edgecolor="0.25",
        linewidth=0.8,
        label="Available endpoints",
        zorder=7,
    )

    # ------------------------------------------------------------
    # Observed endpoint
    # ------------------------------------------------------------

    ax_map.scatter(
        observed_geometry.x,
        observed_geometry.y,
        marker="*",
        s=190,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        label="Observed endpoint",
        zorder=9,
    )

    # ------------------------------------------------------------
    # Map formatting
    # ------------------------------------------------------------

    animal = (
        str(s.iloc[0][
            "individual-local-identifier"
        ])
        if "individual-local-identifier" in s
        else ""
    )

    title = (
        f"{animal} — stratum {stratum_id}"
        if animal
        else f"Stratum {stratum_id}"
    )

    ax_map.set_title(
        title,
        loc="left",
        fontweight="bold",
        fontsize=12,
        pad=8,
    )

    ax_map.set_xlabel(
        "Longitude"
    )

    ax_map.set_ylabel(
        "Latitude"
    )

    ax_map.set_aspect(
        "equal",
        adjustable="box",
    )

    ax_map.legend(
        frameon=False,
        loc="best",
        fontsize=9,
    )

    # ------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------

    pretty = (
        variable_label
        if variable_label is not None
        else output_name.replace(
            "_",
            " ",
        ).title()
    )

    axis_label = (
        f"{pretty} [{unit}]"
        if unit
        else pretty
    )

    cbar = fig.colorbar(
        mappable,
        cax=cax,
    )

    cbar.set_label(
        axis_label,
    )

    # ------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------

    local_values = (
        raster_values[
            np.isfinite(
                raster_values
            )
        ]
    )

    if distribution_kind == "hist":

        # ------------------------------------------------------------
        # Histogram bins
        # ------------------------------------------------------------

        # Restrict the values to the range actually displayed.
        hist_values = local_values[
            np.isfinite(local_values)
            & (local_values >= limits[0])
            & (local_values <= limits[1])
        ]

        if hist_values.size:

            if isinstance(bins, str):

                # Supports NumPy rules such as:
                # "fd", "auto", "scott", "sturges", ...
                edges = np.histogram_bin_edges(
                    hist_values,
                    bins=bins,
                    range=limits,
                )

            else:

                edges = np.linspace(
                    limits[0],
                    limits[1],
                    int(bins) + 1,
                )

            ax_dist.hist(
                hist_values,
                bins=edges,
                density=True,
                histtype="stepfilled",
                alpha=0.14,
                color="0.3",
                edgecolor="0.3",
                linewidth=1,
                label="Local raster",
            )

            ax_dist.hist(
                hist_values,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.4,
                color="0.25",
            )

        ax_dist.set_ylabel(
            "Spatial density"
        )

    elif distribution_kind == "ecdf":

        local_values = np.sort(
            local_values
        )

        cdf = (
            np.arange(
                1,
                len(local_values) + 1,
            )
            / len(local_values)
        )

        ax_dist.step(
            local_values,
            cdf,
            where="post",
            color="0.25",
            linewidth=1.6,
            label="Local raster",
        )

        ax_dist.set_ylabel(
            "Spatial ECDF"
        )

        ax_dist.set_ylim(
            0,
            1,
        )

    else:
        raise ValueError(
            "distribution_kind must be "
            "'hist' or 'ecdf'."
        )

    # ------------------------------------------------------------
    # Available candidate rug
    # ------------------------------------------------------------

    ymin, ymax = ax_dist.get_ylim()

    rug_height = (
        (ymax - ymin) * 0.045
    )

    for value in available_values:

        if not np.isfinite(value):
            continue

        ax_dist.plot(
            [
                value,
                value,
            ],
            [
                ymin,
                ymin + rug_height,
            ],
            color="0.35",
            alpha=available_rug_alpha,
            linewidth=1,
        )

    # ------------------------------------------------------------
    # Observed value
    # ------------------------------------------------------------

    ax_dist.axvline(
        observed_value,
        color="black",
        linewidth=2.4,
        label="Observed endpoint",
        zorder=5,
    )

    # Put a star at the bottom of that line.
    ax_dist.scatter(
        observed_value,
        ymin + rug_height * 0.5,
        marker="*",
        s=110,
        facecolor="white",
        edgecolor="black",
        linewidth=1,
        zorder=6,
    )

    # ------------------------------------------------------------
    # Zero reference
    # ------------------------------------------------------------

    if (
        limits[0] < 0
        < limits[1]
    ):

        ax_dist.axvline(
            0,
            linestyle="--",
            linewidth=0.8,
            color="0.5",
            alpha=0.7,
        )

    # ------------------------------------------------------------
    # Distribution formatting
    # ------------------------------------------------------------

    ax_dist.set_xlim(
        *limits
    )

    ax_dist.set_xlabel(
        axis_label
    )

    ax_dist.set_title(
        "Environmental availability",
        loc="left",
        fontweight="bold",
    )

    ax_dist.spines[
        "top"
    ].set_visible(False)

    ax_dist.spines[
        "right"
    ].set_visible(False)

    ax_dist.grid(
        axis="y",
        alpha=0.2,
        linewidth=0.5,
    )

    ax_dist.legend(
        frameon=False,
        fontsize=9,
    )

    return {
        "figure": fig,
        "map_axis": ax_map,
        "distribution_axis": ax_dist,
        "colorbar_axis": cax,
        "stratum": s,
        "candidate_values": candidate_points,
        "raster": raster,
        "observed_value": observed_value,
        "available_values": available_values,
    }

def _observed_stratum_bearing(
    choices,
    *,
    stratum_id,
):
    """
    Return the geodesic bearing of the observed step in degrees/radians.

    Bearing follows the navigation convention:
        0°   = north
        90°  = east
        180° = south
        270° = west
    """

    import geopandas as gpd
    import numpy as np
    from pyproj import Geod

    s = choices.loc[
        choices["stratum_id"] == stratum_id
    ].copy()

    if s.empty:
        raise ValueError(
            f"Stratum {stratum_id!r} not found."
        )

    observed = s.loc[
        s["used"] == 1
    ]

    if len(observed) != 1:
        raise ValueError(
            "Stratum must contain exactly one observed endpoint."
        )

    if s.crs is None:
        raise ValueError(
            "choices must have a CRS."
        )

    # ------------------------------------------------------------
    # Start point
    # ------------------------------------------------------------

    start = gpd.GeoSeries(
        [s["start_geometry"].iloc[0]],
        crs=s.crs,
    ).to_crs(4326).iloc[0]

    # ------------------------------------------------------------
    # Observed endpoint
    # ------------------------------------------------------------

    endpoint = (
        observed
        .to_crs(4326)
        .geometry
        .iloc[0]
    )

    # ------------------------------------------------------------
    # Geodesic bearing
    # ------------------------------------------------------------

    geod = Geod(
        ellps="WGS84"
    )

    bearing_deg, _, distance_m = (
        geod.inv(
            start.x,
            start.y,
            endpoint.x,
            endpoint.y,
        )
    )

    # Geod returns [-180, 180].
    bearing_deg = (
        bearing_deg + 360.0
    ) % 360.0

    bearing_rad = np.deg2rad(
        bearing_deg
    )

    return {
        "bearing_deg": float(bearing_deg),
        "bearing_rad": float(bearing_rad),
        "distance_m": float(distance_m),
    }

def make_stratum_wind_support_field(
    choices,
    field,
    *,
    stratum_id,
    u_var="u10",
    v_var="v10",
    output_name="wind_support",
    bearing_deg=None,
):
    """
    Construct a dynamic raster of wind support for one SSF stratum.

    By default the reference direction is the observed step bearing.

    Positive:
        tailwind support.

    Negative:
        headwind.

    Parameters
    ----------
    choices
        Canonical SSF choice table.

    field
        xarray Dataset containing eastward and northward wind components.

    stratum_id
        Stratum whose observed travel direction defines the reference bearing.

    u_var, v_var
        Eastward and northward vector-component names.

    output_name
        Name assigned to the derived DataArray.

    bearing_deg
        Optional explicit navigation bearing in degrees clockwise from north.
        If omitted, the observed step bearing is used.
    """

    import numpy as np
    import xarray as xr

    if not isinstance(
        field,
        xr.Dataset,
    ):
        raise TypeError(
            "Wind-support derivation requires an xarray Dataset "
            "containing both u and v components."
        )

    for var in (
        u_var,
        v_var,
    ):
        if var not in field:
            raise KeyError(
                f"{var!r} not found in wind field."
            )

    if bearing_deg is None:

        direction = _observed_stratum_bearing(
            choices,
            stratum_id=stratum_id,
        )

        bearing_deg = direction[
            "bearing_deg"
        ]

        bearing_rad = direction[
            "bearing_rad"
        ]

    else:

        bearing_deg = (
            float(bearing_deg)
            % 360.0
        )

        bearing_rad = np.deg2rad(
            bearing_deg
        )

        direction = {
            "bearing_deg":
                bearing_deg,
            "bearing_rad":
                bearing_rad,
            "distance_m":
                np.nan,
        }

    # ------------------------------------------------------------
    # u = eastward
    # v = northward
    #
    # Projection onto navigation bearing measured clockwise
    # from north.
    # ------------------------------------------------------------

    support = (
        field[u_var]
        * np.sin(bearing_rad)
        +
        field[v_var]
        * np.cos(bearing_rad)
    )

    support = support.rename(
        output_name
    )

    support.attrs.update(
        {
            "long_name":
                "Wind support along observed step direction",
            "units":
                field[u_var].attrs.get(
                    "units",
                    "m s-1",
                ),
            "reference_bearing_deg":
                bearing_deg,
            "positive_direction":
                "tailwind",
            "negative_direction":
                "headwind",
        }
    )

    return support, direction

def plot_stratum_wind_support(
    choices,
    field,
    *,
    stratum_id,
    u_var="u10",
    v_var="v10",
    bearing_deg=None,
    output_name="wind_support",
    **plot_kwargs,
):
    """
    Plot the wind-support landscape for one SSF choice stratum.

    The default reference direction is the observed step direction.
    """

    support, direction = (
        make_stratum_wind_support_field(
            choices,
            field,
            stratum_id=stratum_id,
            u_var=u_var,
            v_var=v_var,
            output_name=output_name,
            bearing_deg=bearing_deg,
        )
    )

    result = plot_stratum_environment(
        choices,
        support,
        stratum_id=stratum_id,
        variable=output_name,
        output_name=output_name,

        variable_label="Wind support",
        unit="m s$^{-1}$",

        cmap="RdBu_r",
        norm_mode="diverging",

        **plot_kwargs,
    )

    result[
        "wind_direction"
    ] = direction

    return result
    
def plot_dynamic_conditions(
    field: xr.Dataset | xr.DataArray,
    *,
    variables: Sequence[str] | Mapping[str, str] | str,
    times,
    bounds: tuple[float, float, float, float] | None = None,
    timezone: str | None = "UTC",
    transforms: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    derived: Mapping[str, Callable[[xr.Dataset], xr.DataArray | np.ndarray]] | None = None,
    extract_kwargs: Mapping[str, Any] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
):
    """Convenience wrapper: extract snapshots, summarize them, and plot them."""
    extract_options = {} if extract_kwargs is None else dict(extract_kwargs)
    plot_options = {} if plot_kwargs is None else dict(plot_kwargs)
    snapshots = extract_dynamic_condition_snapshots(
        field,
        variables=variables,
        times=times,
        bounds=bounds,
        timezone=timezone,
        transforms=transforms,
        derived=derived,
        **extract_options,
    )
    return plot_dynamic_condition_comparison(snapshots, **plot_options)


__all__ = [
    "extract_dynamic_condition_snapshots",
    "summarize_dynamic_conditions",
    "plot_dynamic_condition_comparison",
    "plot_stratum_environment"
    "plot_dynamic_conditions",
    "plot_static_conditions"
]
