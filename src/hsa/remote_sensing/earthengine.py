from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def initialize_earth_engine(project: str | None = None):
    """Initialize Google Earth Engine lazily.

    Keeping this tiny wrapper avoids package-level Earth Engine side effects.
    Examples may call ``initialize_earth_engine(project='...')`` explicitly.
    """

    import ee

    if project is None:
        ee.Initialize()
    else:
        ee.Initialize(project=project)
        ee.data.setCloudApiUserProject(project)
    return ee


def ee_image_to_xarray_stack(image, *, geometry, crs: str, scale: float):
    """Convert an Earth Engine image to a ``band, y, x`` xarray stack.

    This is a thin wrapper around geemap and should remain optional through the
    ``hsa[earthengine]`` dependency group.
    """

    import geemap

    ds = geemap.ee_to_xarray(image, crs=crs, scale=scale, geometry=geometry)
    ds0 = ds.isel(time=0, drop=True) #.rename({"X": "x", "Y": "y"})
    ds0 = ds0.chunk({"y": 1024, "x": 1024}) # TODO: heuristic
    return ds0.to_array(dim="band").transpose("band", "y", "x")


def ee_samples_to_gdf(samples, *, target_crs: str | int | None = None):
    """Convert an Earth Engine sampled FeatureCollection to a GeoDataFrame.

    The Earth Engine object is evaluated with ``getInfo()``, so this helper is
    appropriate for diagnostic samples, not very large exports.
    """

    import geopandas as gpd

    features = samples.getInfo()["features"]
    rows: list[dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties", {}).copy()
        coords = feature.get("geometry", {}).get("coordinates")
        if coords is None:
            continue
        props["Longitude"] = coords[0]
        props["Latitude"] = coords[1]
        rows.append(props)

    df = pd.DataFrame(rows)
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


def spatial_summary(
    image,
    region,
    *,
    scale: float,
    band: str | None = None,
    projection: str = "EPSG:4326",
    num_pixels: int = 4000,
    seed: int = 42,
    target_crs: str | int | None = None,
    include_variogram: bool = True,
) -> tuple[dict[str, Any], Any]:
    """Summarize spatial variability of one Earth Engine image band.

    Returns a summary dictionary and the sampled GeoDataFrame. If
    ``include_variogram=True``, ``scikit-gstat`` is used lazily to estimate an
    empirical variogram and add its effective range to the summary.
    """

    if band is None:
        band = image.bandNames().getInfo()[0]

    samples = image.sample(
        region=region,
        scale=scale,
        projection=projection,
        numPixels=num_pixels,
        seed=seed,
        geometries=True,
    )
    gdf = ee_samples_to_gdf(samples, target_crs=target_crs or projection)
    values = pd.to_numeric(gdf[band], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    summary: dict[str, Any] = {
        "band": band,
        "scale": scale,
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else np.nan,
        "median": float(np.median(values)) if values.size else np.nan,
        "variance": float(np.var(values)) if values.size else np.nan,
        "sd": float(np.std(values)) if values.size else np.nan,
        "min": float(np.min(values)) if values.size else np.nan,
        "max": float(np.max(values)) if values.size else np.nan,
        "effective_range": np.nan,
    }

    if include_variogram and values.size > 2:
        try:
            import skgstat as skg

            complete = gdf.loc[pd.to_numeric(gdf[band], errors="coerce").notna()].copy()
            coords = complete[["x", "y"]].to_numpy(dtype=float)
            vals = pd.to_numeric(complete[band], errors="coerce").to_numpy(dtype=float)
            variogram = skg.Variogram(coords, vals, model="exponential", maxlag="median")
            summary["effective_range"] = float(variogram.describe().get("effective_range", np.nan))
        except Exception as exc:
            summary["variogram_error"] = str(exc)
        return summary, gdf, variogram
    
    return summary, gdf

def temporal_summary(
    image_collection,
    region,
    *,
    band: str | None = None,
    scale: float = 100,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cadence: str = "MS",
    projection: str = "EPSG:4326",
    reducers: Sequence[str] = ("mean", "median", "stdDev", "min", "max"),
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    import ee
    import time

    def _log(msg: str):
        if verbose:
            print(msg, flush=True)

    t_start_total = time.time()

    _log("Preparing temporal summary...")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    edges = pd.date_range(start=start_ts, end=end_ts, freq=cadence)

    if len(edges) == 0 or edges[0] != start_ts:
        edges = edges.insert(0, start_ts)
    if edges[-1] < end_ts:
        edges = edges.append(pd.DatetimeIndex([end_ts]))

    n_periods = len(edges) - 1
    _log(f"Prepared {n_periods} periods from {start_ts.date()} to {end_ts.date()}.")

    if band is None:
        _log("No band supplied; retrieving first band name from Earth Engine...")
        band = image_collection.first().bandNames().getInfo()[0]

    _log(f"Using band: {band}")
    _log(f"Reducers: {', '.join(reducers)}")

    reducer_map = {
        "mean": ee.Reducer.mean(),
        "median": ee.Reducer.median(),
        "stdDev": ee.Reducer.stdDev(),
        "min": ee.Reducer.min(),
        "max": ee.Reducer.max(),
        "variance": ee.Reducer.variance(),
    }

    unknown = [name for name in reducers if name not in reducer_map]
    if unknown:
        raise ValueError(
            f"Unsupported reducers: {unknown}. Choose from {sorted(reducer_map)}."
        )

    reducer = reducer_map[reducers[0]]
    for name in reducers[1:]:
        reducer = reducer.combine(reducer_map[name], sharedInputs=True)

    _log("Building server-side period FeatureCollection...")

    period_features = [
        ee.Feature(
            None,
            {
                "period_start": pd.Timestamp(t0).isoformat(),
                "period_end": pd.Timestamp(t1).isoformat(),
            },
        )
        for t0, t1 in zip(edges[:-1], edges[1:])
    ]

    periods = ee.FeatureCollection(period_features)
    collection = image_collection.select(band)

    def summarize_period(feature):
        t0 = ee.Date(feature.get("period_start"))
        t1 = ee.Date(feature.get("period_end"))

        subset = collection.filterDate(t0, t1)
        n_images = subset.size()

        stats = ee.Dictionary(
            ee.Algorithms.If(
                n_images.gt(0),
                subset.mean().reduceRegion(
                    reducer=reducer,
                    geometry=region,
                    scale=scale,
                    crs=projection,
                    maxPixels=1e9,
                    bestEffort=True,
                ),
                ee.Dictionary(),
            )
        )

        props = (
            feature
            .toDictionary()
            .combine(stats, True)
            .set("n_images", n_images)
        )

        return ee.Feature(None, props)

    _log(
        "Submitting Earth Engine request. "
        "This may take a while; Python cannot show per-period progress during getInfo()."
    )

    t0 = time.time()
    features = periods.map(summarize_period).getInfo()["features"]
    _log(f"Earth Engine request finished in {(time.time() - t0):.1f} seconds.")

    _log("Parsing returned period summaries...")

    rows = []
    for i, feature in enumerate(features, start=1):
        props = feature.get("properties", {})

        row = {
            "period_start": pd.Timestamp(props.get("period_start")),
            "period_end": pd.Timestamp(props.get("period_end")),
            "n_images": int(props.get("n_images", 0)),
        }

        for name in reducers:
            row[name] = props.get(f"{band}_{name}", props.get(name, np.nan))

        rows.append(row)

        if verbose:
            period_start = row["period_start"].date()
            period_end = row["period_end"].date()
            n_images = row["n_images"]
            print(
                f"Parsed {i}/{n_periods}: {period_start} to {period_end}, "
                f"{n_images} images",
                flush=True,
            )

    _log("Formatting output tables...")

    wide = (
        pd.DataFrame(rows)
        .sort_values("period_start")
        .reset_index(drop=True)
    )

    wide["period_mid"] = (
        wide["period_start"]
        + (wide["period_end"] - wide["period_start"]) / 2
    )

    wide["duration_days"] = (
        wide["period_end"] - wide["period_start"]
    ).dt.total_seconds() / 86400.0

    for name in reducers:
        wide[name] = pd.to_numeric(wide[name], errors="coerce")
        wide[f"{name}_diff"] = wide[name].diff()

        previous = wide[name].shift(1)
        wide[f"{name}_pct_change"] = np.where(
            previous != 0,
            (wide[name] - previous) / previous,
            np.nan,
        )

    value_columns = list(reducers)

    long = wide.melt(
        id_vars=[
            "period_start",
            "period_end",
            "period_mid",
            "duration_days",
            "n_images",
        ],
        value_vars=value_columns,
        var_name="statistic",
        value_name="value",
    )

    _log(f"Done. Total runtime: {(time.time() - t_start_total):.1f} seconds.")

    return wide, long
