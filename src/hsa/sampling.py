from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def get_availability_domain(
    used: gpd.GeoDataFrame,
    *,
    id_col: str | None = None,
    estimator: str = "mcp",
    quantile: float = 0.95,
    min_points: int = 5,
    keep_columns: bool = True,
) -> gpd.GeoDataFrame:
    """
    Create availability domain(s) from used locations.

    If id_col is None, returns one population-level domain.
    If id_col is supplied, returns one domain per individual.

    Parameters
    ----------
    used:
        GeoDataFrame of used locations.
    id_col:
        Optional individual ID column. If supplied, availability is estimated
        separately per individual.
    estimator:
        Currently only "mcp".
    quantile:
        Quantile used for centroid-trimmed MCP.
    min_points:
        Minimum number of points required per domain.
    keep_columns:
        If True and id_col is supplied, include n_points and quantile metadata.

    Returns
    -------
    GeoDataFrame
        Availability polygons. If id_col is supplied, one row per individual.
    """

    if estimator.lower() != "mcp":
        raise ValueError(
            f"Unsupported estimator: {estimator!r}. Currently only 'mcp' is implemented."
        )

    if used.crs is None:
        raise ValueError("used.crs is None; set a CRS before defining availability.")

    if id_col is not None and id_col not in used.columns:
        raise ValueError(f"id_col={id_col!r} not found in used.columns.")

    original_crs = used.crs

    def _single_domain(g: gpd.GeoDataFrame):
        if len(g) < min_points:
            return None

        g_metric = g if not original_crs.is_geographic else g.to_crs(g.estimate_utm_crs())

        centroid = g_metric.geometry.union_all().centroid

        tmp = g_metric.copy()
        tmp["distance_to_centroid"] = tmp.geometry.distance(centroid)

        keep = (
            tmp["distance_to_centroid"]
            <= tmp["distance_to_centroid"].quantile(quantile)
        )

        kept = tmp.loc[keep]

        if len(kept) < min_points:
            return None

        hull = kept.geometry.union_all().convex_hull
        area_km2 = hull.area / 1_000_000

        if hull.is_empty:
            return None

        return gpd.GeoDataFrame(
            {
                "n_points": [len(g)],
                "n_points_used_for_hull": [len(kept)],
                "quantile": [quantile],
                "estimator": [estimator],
                "area_km²": [area_km2],
            },
            geometry=[hull],
            crs=g_metric.crs,
        ).to_crs(original_crs)

    if id_col is None:
        domain = _single_domain(used)
        if domain is None:
            raise ValueError(
                f"Could not estimate availability domain: fewer than {min_points} valid points."
            )
        return domain

    domains = []

    for individual_id, group in used.groupby(id_col):
        domain = _single_domain(group)

        if domain is None:
            continue

        domain[id_col] = individual_id
        domains.append(domain)

    if not domains:
        raise ValueError(
            f"Could not estimate any individual domains. "
            f"Check min_points={min_points}, quantile={quantile}, and id_col={id_col!r}."
        )

    out = pd.concat(domains, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=original_crs)

    cols = [id_col] + [c for c in out.columns if c != id_col]
    out = out[cols]

    if not keep_columns:
        out = out[[id_col, "geometry"]]

    return out


def sample_available_points(
    domain: gpd.GeoDataFrame,
    n: int | None = None,
    used: gpd.GeoDataFrame | None = None,
    *,
    id_col: str | None = None,
    n_per_used: int | None = None,
    seed: int = 42,
    timestamp_col: str = "Timestamp",
    keep_domain_columns: bool = False,
) -> gpd.GeoDataFrame:
    """
    Sample available points and optionally append used points.

    If id_col is None:
        sample n points from the whole domain.

    If id_col is supplied:
        domain must contain one row per individual/domain.
        available points are sampled separately inside each individual's domain.
        If used is supplied, used points are matched by id_col and appended.
    """

    if domain.crs is None:
        raise ValueError("domain.crs is None; set a CRS before sampling available points.")

    if id_col is not None and id_col not in domain.columns:
        raise ValueError(f"id_col={id_col!r} not found in domain.columns.")

    if used is not None:
        if used.crs is None:
            raise ValueError("used.crs is None; set a CRS before appending used points.")
        if used.crs != domain.crs:
            used = used.to_crs(domain.crs)
        if id_col is not None and id_col not in used.columns:
            raise ValueError(f"id_col={id_col!r} not found in used.columns.")

    if id_col is None:
        if n is None:
            raise ValueError("n must be supplied when id_col is None.")

        available = gpd.GeoDataFrame(
            geometry=domain.sample_points(n, rng=seed).explode(index_parts=True),
            crs=domain.crs,
        ).reset_index(drop=True)

        if used is not None:
            available[timestamp_col] = pd.Series(
                pd.NaT,
                index=available.index,
                dtype=used[timestamp_col].dtype,
            )
        else:
            available[timestamp_col] = pd.NaT

        available["used"] = False

        if used is None:
            return available

        used_min = used[[timestamp_col, "geometry"]].copy()
        used_min["used"] = True

        samples = pd.concat([used_min, available], ignore_index=True)
        return gpd.GeoDataFrame(samples, geometry="geometry", crs=domain.crs)

    frames = []

    for i, domain_row in domain.reset_index(drop=True).iterrows():
        individual_id = domain_row[id_col]

        domain_i = gpd.GeoDataFrame(
            [domain_row],
            geometry="geometry",
            crs=domain.crs,
        )

        used_i = None
        if used is not None:
            used_i = used.loc[used[id_col] == individual_id].copy()

        if n_per_used is not None:
            if used_i is None:
                raise ValueError("n_per_used requires used to be supplied.")
            n_i = len(used_i) * n_per_used
        elif n is not None:
            n_i = n
        else:
            raise ValueError("Supply either n or n_per_used.")

        if n_i <= 0:
            continue

        available_i = gpd.GeoDataFrame(
            geometry=domain_i.sample_points(n_i, rng=seed + i).explode(index_parts=True),
            crs=domain.crs,
        ).reset_index(drop=True)

        available_i[id_col] = individual_id
        if used_i is not None and not used_i.empty:
            available_i[timestamp_col] = pd.Series(
                pd.NaT,
                index=available_i.index,
                dtype=used_i[timestamp_col].dtype,
            )
        else:
            available_i[timestamp_col] = pd.NaT
        available_i["used"] = False

        if keep_domain_columns:
            for col in domain.columns:
                if col not in {id_col, "geometry"}:
                    available_i[col] = domain_row[col]

        frames.append(available_i)

        if used_i is not None and not used_i.empty:
            used_cols = [id_col, timestamp_col, "geometry"]
            used_min = used_i[used_cols].copy()
            used_min["used"] = True
            frames.append(used_min)

    if not frames:
        raise ValueError("No available points were sampled. Check domains, n, and n_per_used.")

    samples = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(samples, geometry="geometry", crs=domain.crs)


def _as_raster_dataarray(env: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """Normalize a raster stack to one ``DataArray(band, y, x)``.

    ``xr.open_zarr`` returns a Dataset even when the Zarr store originated from
    a single DataArray. Accept that common representation automatically, while
    refusing ambiguous multi-variable Datasets.
    """
    if isinstance(env, xr.DataArray):
        raster = env
    elif isinstance(env, xr.Dataset):
        candidates = [
            name
            for name, variable in env.data_vars.items()
            if {"band", "y", "x"}.issubset(variable.dims)
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Raster Dataset must contain exactly one data variable with "
                "dimensions ('band', 'y', 'x'). Found compatible variables: "
                f"{candidates}. Select the desired variable explicitly before sampling."
            )
        raster = env[candidates[0]]
    else:
        raise TypeError("env must be an xarray.DataArray or xarray.Dataset.")

    missing_dims = [dim for dim in ("band", "y", "x") if dim not in raster.dims]
    if missing_dims:
        raise ValueError(
            "Raster stack must contain dimensions ('band', 'y', 'x'); "
            f"missing {missing_dims}."
        )
    return raster


def infer_resolution(env: xr.DataArray | xr.Dataset) -> tuple[float, float]:
    """Infer absolute x/y resolution from a regular xarray grid."""
    raster = _as_raster_dataarray(env)
    x = raster["x"].values
    y = raster["y"].values
    return float(np.abs(np.diff(x)).mean()), float(np.abs(np.diff(y)).mean())


def aggregate_env(
    env: xr.DataArray | xr.Dataset,
    target_resolution: float,
    *,
    reducer: str = "mean",
) -> xr.DataArray:
    """Aggregate an environmental stack to an integer multiple of its native resolution."""
    raster = _as_raster_dataarray(env)
    dx, dy = infer_resolution(raster)
    fx = target_resolution / dx
    fy = target_resolution / dy

    if not float(fx).is_integer() or not float(fy).is_integer():
        raise ValueError(
            f"target_resolution={target_resolution} is not an integer multiple of "
            f"native resolution ({dx:.3f}, {dy:.3f})."
        )

    fx = int(round(fx))
    fy = int(round(fy))
    if fx == 1 and fy == 1:
        return raster

    coarsened = raster.coarsen(x=fx, y=fy, boundary="trim")
    reducers = {
        "mean": coarsened.mean,
        "median": coarsened.median,
        "max": coarsened.max,
        "min": coarsened.min,
    }
    if reducer not in reducers:
        raise ValueError(f"Unsupported reducer: {reducer!r}")

    out = reducers[reducer]()
    return out.transpose(*[d for d in ("band", "y", "x") if d in out.dims])


def sample_raster_stack(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray | xr.Dataset,
    bands=None,
    *,
    target_resolution: float | None = None,
    reducer: str = "mean",
    dtype: str = "float32",
    id_cols: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Sample an xarray raster stack at point locations using nearest-neighbour lookup."""
    raster = _as_raster_dataarray(env)

    if bands is not None:
        raster = raster.sel(band=bands)

    if target_resolution is not None:
        raster = aggregate_env(
            raster,
            target_resolution=target_resolution,
            reducer=reducer,
        )

    try:
        env_crs = raster.rio.crs
    except Exception:
        env_crs = None

    if env_crs is not None and samples.crs is not None and samples.crs != env_crs:
        samples = samples.to_crs(env_crs)

    raster = raster.transpose("band", "y", "x")

    xs = xr.DataArray(samples.geometry.x.to_numpy(), dims="points", name="x")
    ys = xr.DataArray(samples.geometry.y.to_numpy(), dims="points", name="y")

    sampled = raster.sel(x=xs, y=ys, method="nearest").transpose("band", "points")

    arr = sampled.data
    arr = arr.compute() if hasattr(arr, "compute") else np.asarray(arr)
    arr = np.asarray(arr, dtype=dtype)

    sampled_bands = [str(band) for band in sampled["band"].values]
    out = pd.DataFrame(arr.T, columns=sampled_bands, index=samples.index)

    out["x"] = xs.values
    out["y"] = ys.values

    for column in ("used", "Timestamp"):
        if column in samples.columns:
            out[column] = samples[column].to_numpy()

    if id_cols is not None:
        if isinstance(id_cols, str):
            id_cols = [id_cols]

        missing = [col for col in id_cols if col not in samples.columns]
        if missing:
            raise ValueError(f"id_cols not found in samples.columns: {missing}")

        for column in id_cols:
            out[column] = samples[column].to_numpy()

    return out


def sample_raster_stack_multiscale(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray | xr.Dataset,
    *,
    target_resolutions: list[float] | tuple[float, ...],
    reducer: str = "mean",
    dtype: str = "float32",
) -> tuple[pd.DataFrame, dict[str, xr.DataArray]]:
    """Sample the same stack at multiple spatial resolutions.

    Returns the sampled dataframe and the aggregated stacks used for prediction.
    Column names are suffixed with ``_30m``, ``_90m`` and so forth.
    """
    raster = _as_raster_dataarray(env)
    frames: list[pd.DataFrame] = []
    env_dict: dict[str, xr.DataArray] = {}

    for resolution in target_resolutions:
        suffix = f"{int(resolution)}m" if float(resolution).is_integer() else f"{resolution}m"
        env_res = aggregate_env(
            raster,
            target_resolution=float(resolution),
            reducer=reducer,
        )
        env_dict[suffix] = env_res
        sampled = sample_raster_stack(samples, env_res, dtype=dtype)
        value_columns = [
            c
            for c in sampled.columns
            if c not in {"x", "y", "used", "Timestamp"}
        ]
        sampled = sampled.rename(
            columns={c: f"{c}_{suffix}" for c in value_columns}
        )
        frames.append(
            sampled.drop(
                columns=[
                    c
                    for c in ("x", "y", "used", "Timestamp")
                    if c in sampled.columns
                ]
            )
        )

    out = pd.concat(frames, axis=1)
    out["x"] = samples.geometry.x.to_numpy()
    out["y"] = samples.geometry.y.to_numpy()
    for column in ("used", "Timestamp"):
        if column in samples.columns:
            out[column] = samples[column].to_numpy()

    return out, env_dict
