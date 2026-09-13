"""Environmental annotation helpers for SSF candidate endpoints."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod

from hsa.sampling import sample_raster_stack
from hsa._time import require_timezone_aware


def check_raster_coverage(
    points: gpd.GeoDataFrame,
    env: xr.DataArray,
) -> pd.Series:
    """Return whether each point lies inside the raster coordinate extent."""
    transformed = points
    try:
        env_crs = env.rio.crs
    except Exception:
        env_crs = None

    if (
        env_crs is not None
        and transformed.crs is not None
        and transformed.crs != env_crs
    ):
        transformed = transformed.to_crs(env_crs)

    xmin, xmax = sorted(
        (float(env["x"].min()), float(env["x"].max()))
    )
    ymin, ymax = sorted(
        (float(env["y"].min()), float(env["y"].max()))
    )
    x = transformed.geometry.x.to_numpy(dtype=float)
    y = transformed.geometry.y.to_numpy(dtype=float)
    inside = (
        (x >= xmin)
        & (x <= xmax)
        & (y >= ymin)
        & (y <= ymax)
    )
    return pd.Series(
        inside,
        index=points.index,
        name="inside_env",
    )


def sample_static_covariates_batched(
    points: gpd.GeoDataFrame,
    env: xr.DataArray,
    *,
    bands: Sequence[str],
    batch_size: int = 100_000,
    id_cols: Sequence[str] | str | None = None,
    require_inside: bool = True,
) -> pd.DataFrame:
    """Sample endpoint raster covariates in bounded row batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    bands = list(dict.fromkeys(bands))
    if not bands:
        raise ValueError("At least one raster band is required.")

    if "band" not in env.coords:
        raise ValueError("env must expose a 'band' coordinate.")
    available_bands = set(map(str, env["band"].values))
    missing_bands = sorted(set(bands).difference(available_bands))
    if missing_bands:
        raise KeyError(f"Raster bands not found: {missing_bands}")

    if require_inside:
        inside = check_raster_coverage(points, env)
        if not inside.all():
            n_out = int((~inside).sum())
            raise ValueError(
                f"{n_out:,} SSF candidate endpoints fall outside the raster extent. "
                "Regenerate/redraw candidates inside the valid availability domain "
                "rather than snapping them to the raster edge."
            )

    frames: list[pd.DataFrame] = []
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        batch = points.iloc[start:stop].copy()
        sampled = sample_raster_stack(
            batch,
            env,
            bands=bands,
            id_cols=id_cols,
        )
        if isinstance(sampled, tuple):
            sampled = sampled[0]
        sampled = sampled.copy()
        sampled.index = batch.index
        frames.append(sampled[bands])

    return pd.concat(frames).sort_index()


def annotate_static_covariates(
    choices: gpd.GeoDataFrame,
    env: xr.DataArray,
    *,
    bands: Sequence[str],
    batch_size: int = 100_000,
    require_inside: bool = True,
) -> gpd.GeoDataFrame:
    """Return choices with static endpoint covariates attached."""
    sampled = sample_static_covariates_batched(
        choices,
        env,
        bands=bands,
        batch_size=batch_size,
        require_inside=require_inside,
    )
    out = choices.copy()
    for band in bands:
        out[band] = sampled[band]
    return out


def _compute_with_retry(
    obj,
    *,
    retries: int = 5,
    initial_delay: float = 1.0,
):
    delay = float(initial_delay)
    last_error = None
    for attempt in range(retries):
        try:
            return obj.compute()
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(
        f"Failed to compute remote xarray slab after {retries} attempts."
    ) from last_error


def _match_longitude_convention(
    longitude: np.ndarray,
    field_longitude: xr.DataArray,
) -> np.ndarray:
    """Convert geographic longitudes to the grid's -180..180 or 0..360 convention."""
    longitude = np.asarray(longitude, dtype=float)
    lon_min = float(field_longitude.min())
    lon_max = float(field_longitude.max())
    if lon_min >= 0 and lon_max > 180:
        return longitude % 360.0
    return ((longitude + 180.0) % 360.0) - 180.0


def _normalize_dynamic_variables(
    variables: Sequence[str] | Mapping[str, str] | str,
) -> dict[str, str]:
    """Return a source-variable -> output-column mapping."""
    if isinstance(variables, str):
        mapping = {variables: variables}
    elif isinstance(variables, Mapping):
        mapping = {str(source): str(output) for source, output in variables.items()}
    else:
        mapping = {str(name): str(name) for name in variables}

    if not mapping:
        raise ValueError("At least one dynamic variable is required.")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Dynamic output column names must be unique.")
    return mapping


def sample_dynamic_covariates_at_points(
    points: gpd.GeoDataFrame,
    field: xr.Dataset | xr.DataArray,
    *,
    variables: Sequence[str] | Mapping[str, str] | str,
    time_col: str = "end_time",
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    time_name: str = "valid_time",
    spatial_margin: float = 0.2,
    time_margin: str | pd.Timedelta = "1h",
    method: str = "linear",
    batch_freq: str | None = "M",
    retries: int = 5,
    initial_delay: float = 1.0,
    transforms: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    dtype: str | np.dtype = "float32",
) -> gpd.GeoDataFrame:
    """Sample time-varying gridded covariates at every SSF candidate endpoint.

    Parameters
    ----------
    points
        Candidate endpoints. Geometry may use any CRS; sampling is performed in
        geographic longitude/latitude after reprojection to EPSG:4326.
    field
        Xarray Dataset, or a single DataArray, with time/latitude/longitude
        coordinates. Remote or Dask-backed arrays are supported by loading only
        bounded spatiotemporal slabs with retry logic.
    variables
        Variable names to sample. A mapping renames sampled variables from
        ``source_name -> output_column``.
    transforms
        Optional per-source transformations applied after interpolation. This is
        useful for unit/sign conventions, e.g. converting an hourly accumulated
        ERA5 flux in J m-2 to an upward-positive mean flux in W m-2 with
        ``{"sshf": lambda x: -x / 3600.0}``.
    batch_freq
        Pandas period frequency used to group queries into bounded time slabs.
        The monthly default works well for multi-year ERA5 data. Set to ``None``
        to query one slab for all points.

    Notes
    -----
    The sampler is intentionally agnostic about variable semantics. Instantaneous,
    mean and accumulated fields can all be sampled, but accumulated variables
    should be converted/interpreted using their documented accumulation period.
    """
    if points.crs is None:
        raise ValueError("points.crs is None.")
    if time_col not in points:
        raise KeyError(f"{time_col!r} not found in points.")
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'.")
    if retries <= 0:
        raise ValueError("retries must be positive.")
    if spatial_margin < 0:
        raise ValueError("spatial_margin must be non-negative.")

    variable_map = _normalize_dynamic_variables(variables)
    source_variables = list(variable_map)
    transforms = {} if transforms is None else dict(transforms)
    unknown_transforms = sorted(set(transforms).difference(source_variables))
    if unknown_transforms:
        raise KeyError(
            "Transforms were supplied for variables that are not sampled: "
            f"{unknown_transforms}"
        )

    if isinstance(field, xr.DataArray):
        if field.name is None:
            if len(source_variables) != 1:
                raise ValueError(
                    "An unnamed DataArray can only be sampled when one variable "
                    "is requested."
                )
            field = field.to_dataset(name=source_variables[0])
        else:
            field = field.to_dataset()
    elif not isinstance(field, xr.Dataset):
        raise TypeError("field must be an xarray Dataset or DataArray.")

    for name in (longitude_name, latitude_name, time_name):
        if name not in field.coords:
            raise KeyError(f"{name!r} not found in dynamic-field coordinates.")
    missing_variables = sorted(set(source_variables).difference(field.data_vars))
    if missing_variables:
        raise KeyError(f"Dynamic variables not found: {missing_variables}")

    out = points.copy()
    ll = out.to_crs(4326)
    ll["_lon"] = _match_longitude_convention(
        ll.geometry.x.to_numpy(dtype=float),
        field[longitude_name],
    )
    ll["_lat"] = ll.geometry.y.to_numpy(dtype=float)
    ll["_time"] = require_timezone_aware(
        ll[time_col],
        name=time_col,
        to_utc=True,
    ).dt.tz_localize(None)
    if ll["_time"].isna().any():
        raise ValueError(f"{time_col!r} contains invalid timestamps.")

    if batch_freq is None:
        ll["_batch"] = 0
    else:
        ll["_batch"] = ll["_time"].dt.to_period(batch_freq)

    for output_col in variable_map.values():
        ll[output_col] = np.nan

    dt_margin = pd.Timedelta(time_margin)
    lon_coord = field[longitude_name]
    lat_coord = field[latitude_name]
    lon_min = float(lon_coord.min())
    lon_max = float(lon_coord.max())
    lat_min = float(lat_coord.min())
    lat_max = float(lat_coord.max())

    descending_lon = float(lon_coord.isel({longitude_name: 0})) > float(
        lon_coord.isel({longitude_name: -1})
    )
    descending_lat = float(lat_coord.isel({latitude_name: 0})) > float(
        lat_coord.isel({latitude_name: -1})
    )

    for _, index in ll.groupby("_batch", sort=True).groups.items():
        batch = ll.loc[index]
        xmin = max(lon_min, float(batch["_lon"].min()) - spatial_margin)
        xmax = min(lon_max, float(batch["_lon"].max()) + spatial_margin)
        ymin = max(lat_min, float(batch["_lat"].min()) - spatial_margin)
        ymax = min(lat_max, float(batch["_lat"].max()) + spatial_margin)
        tmin = batch["_time"].min() - dt_margin
        tmax = batch["_time"].max() + dt_margin

        lon_slice = slice(xmax, xmin) if descending_lon else slice(xmin, xmax)
        lat_slice = slice(ymax, ymin) if descending_lat else slice(ymin, ymax)

        slab = field[source_variables].sel(
            {
                latitude_name: lat_slice,
                longitude_name: lon_slice,
                time_name: slice(tmin, tmax),
            }
        )
        slab = _compute_with_retry(
            slab,
            retries=retries,
            initial_delay=initial_delay,
        )

        n = len(batch)
        coords = {"point": np.arange(n)}
        longitude = xr.DataArray(
            batch["_lon"].to_numpy(),
            dims="point",
            coords=coords,
        )
        latitude = xr.DataArray(
            batch["_lat"].to_numpy(),
            dims="point",
            coords=coords,
        )
        timestamp = xr.DataArray(
            batch["_time"].to_numpy(),
            dims="point",
            coords=coords,
        )
        indexers = {
            longitude_name: longitude,
            latitude_name: latitude,
            time_name: timestamp,
        }
        if method == "linear":
            sampled = slab.interp(indexers)
        else:
            sampled = slab.sel(indexers, method="nearest")

        for source_var, output_col in variable_map.items():
            values = np.asarray(sampled[source_var].values)
            if source_var in transforms:
                values = np.asarray(transforms[source_var](values))
            if values.shape != (n,):
                values = np.asarray(values).reshape(n)
            ll.loc[index, output_col] = values.astype(dtype, copy=False)

    for output_col in variable_map.values():
        out[output_col] = ll[output_col].to_numpy(dtype=dtype)
    return out


def sample_dynamic_vectors_at_points(
    points: gpd.GeoDataFrame,
    field: xr.Dataset,
    *,
    time_col: str = "end_time",
    u_var: str = "u10",
    v_var: str = "v10",
    longitude_name: str = "longitude",
    latitude_name: str = "latitude",
    time_name: str = "valid_time",
    spatial_margin: float = 0.2,
    time_margin: str | pd.Timedelta = "1h",
    method: str = "linear",
    batch_freq: str | None = "M",
    retries: int = 5,
    initial_delay: float = 1.0,
    speed_col: str = "wind_speed",
) -> gpd.GeoDataFrame:
    """Sample a gridded east/north vector field at candidate endpoints.

    This is a backwards-compatible convenience wrapper around
    :func:`sample_dynamic_covariates_at_points` that additionally derives vector
    magnitude. The default coordinate/variable names match ERA5-Land 10-m wind.
    """
    out = sample_dynamic_covariates_at_points(
        points,
        field,
        variables=[u_var, v_var],
        time_col=time_col,
        longitude_name=longitude_name,
        latitude_name=latitude_name,
        time_name=time_name,
        spatial_margin=spatial_margin,
        time_margin=time_margin,
        method=method,
        batch_freq=batch_freq,
        retries=retries,
        initial_delay=initial_delay,
    )
    out[speed_col] = np.hypot(out[u_var], out[v_var]).astype("float32")
    return out


def add_vector_support_covariates(
    choices: gpd.GeoDataFrame,
    *,
    start_geometry_col: str = "start_geometry",
    u_col: str = "u10",
    v_col: str = "v10",
    speed_col: str = "wind_speed",
    prefix: str = "wind",
) -> gpd.GeoDataFrame:
    """Project an east/north vector field onto each candidate's geodesic bearing."""
    if choices.crs is None:
        raise ValueError("choices.crs is None.")
    for column in (start_geometry_col, u_col, v_col):
        if column not in choices:
            raise KeyError(f"{column!r} not found in choices.")

    start_ll = gpd.GeoSeries(
        choices[start_geometry_col],
        index=choices.index,
        crs=choices.crs,
    ).to_crs(4326)
    end_ll = choices.geometry.to_crs(4326)
    geod = Geod(ellps="WGS84")
    azimuth_deg, _, geodesic_length = geod.inv(
        start_ll.x.to_numpy(),
        start_ll.y.to_numpy(),
        end_ll.x.to_numpy(),
        end_ll.y.to_numpy(),
    )
    bearing = np.deg2rad(azimuth_deg)
    u = choices[u_col].to_numpy(dtype=float)
    v = choices[v_col].to_numpy(dtype=float)

    support_name = f"{prefix}_support"
    alignment_name = f"{prefix}_alignment"
    if prefix == "wind":
        cross_name = "crosswind"
        abs_cross_name = "abs_crosswind"
    else:
        cross_name = f"{prefix}_cross"
        abs_cross_name = f"abs_{prefix}_cross"

    out = choices.copy()
    out["bearing_rad"] = bearing
    out["geodesic_step_length"] = geodesic_length
    out[support_name] = u * np.sin(bearing) + v * np.cos(bearing)
    out[cross_name] = u * np.cos(bearing) - v * np.sin(bearing)
    out[abs_cross_name] = np.abs(out[cross_name])

    if speed_col in out:
        speed = out[speed_col].to_numpy(dtype=float)
    else:
        speed = np.hypot(u, v)
        out[speed_col] = speed

    out[alignment_name] = np.divide(
        out[support_name].to_numpy(dtype=float),
        speed,
        out=np.full(len(out), np.nan, dtype=float),
        where=speed > 0,
    )
    return out


def add_movement_terms(
    choices: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add common movement terms retained for later integrated SSF models."""
    out = choices.copy()
    if "step_length" in out:
        if (out["step_length"] <= 0).any():
            raise ValueError("step_length must be positive before taking logs.")
        out["log_step_length"] = np.log(out["step_length"])
    if "turn_angle" in out:
        out["cos_turn_angle"] = np.cos(out["turn_angle"])
        out["sin_turn_angle"] = np.sin(out["turn_angle"])
    return out


sample_era5_at_points = sample_dynamic_covariates_at_points
sample_era5_wind_at_points = sample_dynamic_vectors_at_points


__all__ = [
    "check_raster_coverage",
    "sample_static_covariates_batched",
    "annotate_static_covariates",
    "sample_dynamic_covariates_at_points",
    "sample_dynamic_vectors_at_points",
    "sample_era5_at_points",
    "sample_era5_wind_at_points",
    "add_vector_support_covariates",
    "add_movement_terms",
]
