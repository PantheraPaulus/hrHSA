# Dynamic environmental condition plots

Time-varying predictors are often easier to understand before they enter an SSF. `hsa.ssf` therefore provides a quick-look workflow for asking questions such as:

> How did the spatial distribution of uplift and sensible heat flux change between 09:00 and 12:00 on the same winter day?

The diagnostic operates on the gridded dynamic field itself rather than on already-sampled SSF alternatives. This makes it useful for checking variable semantics, diurnal development, spatial heterogeneity, and transformations before model fitting.

## Analysis-object workflow

```python
conditions = ssf.plot_dynamic_conditions(
    era5,
    variables={
        "upward_potential": "upward_potential",
        "sshf": "thermal_flux_upward",
    },
    times=[
        "2025-01-15 09:00",
        "2025-01-15 12:00",
    ],
    timezone="Europe/Madrid",
    transforms={
        "sshf": lambda x: -x / 3600.0,
    },
    plot_kwargs={
        "kind": "both",
        "distribution_kind": "hist",
        "variable_labels": {
            "upward_potential": "Upward potential",
            "thermal_flux_upward": "Upward sensible heat flux",
        },
        "units": {
            "thermal_flux_upward": "W m-2",
        },
    },
)
```

`kind="both"` creates one row per variable. Each requested time gets a spatial map, followed by a distribution panel using the same value range. This makes a 09:00-versus-12:00 comparison visually direct rather than requiring manual xarray slicing and plotting.

The returned object contains:

```python
conditions["figure"]
conditions["axes"]
conditions["summary"]
conditions["snapshots"]
```

The summary reports the spatial mean, standard deviation, 5/25/50/75/95% quantiles, fraction of the mapped domain above zero, and the number of finite grid cells at each requested time.

## Spatial domain

The object-level method defaults to the geographic extent of the relocations and expands it by `spatial_margin=0.25` degrees. This prevents a remote global ERA5 store from being loaded merely to make a diagnostic plot.

Use all SSF alternatives instead with:

```python
ssf.plot_dynamic_conditions(
    era5,
    variables=["upward_potential", "sshf"],
    times=[...],
    domain="choices",
)
```

or supply explicit geographic bounds:

```python
ssf.plot_dynamic_conditions(
    era5,
    variables=["upward_potential", "sshf"],
    times=[...],
    domain=(-6.0, 40.0, 3.0, 44.5),  # west, south, east, north
)
```

Ordinary study regions that cross the Greenwich meridian are handled correctly even when the backing field stores longitudes in the ERA5-style `0..360` convention. A region such as `(-6, ..., 3, ...)` is therefore no longer mistaken for a dateline crossing.

True domains spanning the ±180-degree dateline are still outside the quick-look API and should be split into two regions.

## Fast lower-resolution quick looks

A diagnostic plot usually does not need every native ERA5 grid cell. `resolution` provides a cheap spatial decimation target in geographic degrees:

```python
ssf.plot_dynamic_conditions(
    era5,
    variables=["upward_potential", "sshf"],
    times=["2025-01-15 09:00", "2025-01-15 12:00"],
    timezone="Europe/Madrid",
    domain=(-6.0, 40.0, 3.0, 44.5),
    resolution=0.5,
)
```

For example, a 0.1-degree field with `resolution=0.5` is sampled approximately every fifth cell in each spatial dimension before the requested time slices are computed. This is **decimation**, not conservative spatial averaging, and is intended for exploratory visualization rather than production-scale spatial aggregation.

Alternatively, cap the approximate plotted grid size directly:

```python
ssf.plot_dynamic_conditions(
    era5,
    variables="sshf",
    times=[...],
    max_cells=50_000,
)
```

`resolution` and `max_cells` can be combined. The returned snapshot attributes record the applied latitude/longitude strides and final spatial cell count.

## Local-clock versus UTC comparisons

ERA5-style fields normally expose a UTC time coordinate. Naive timestamps passed to the quick-look API are first localized to `timezone` and then converted to UTC for lookup. This is important for ecological questions framed in local solar/clock time.

For example:

```python
times=["2025-01-15 09:00", "2025-01-15 12:00"]
timezone="Europe/Madrid"
```

compares the two local clock times while querying the corresponding UTC slices. If `timezone=None`, timestamps must already contain timezone information.

The returned snapshots keep both `requested_time_label` and `matched_time_utc`, so nearest-time selection remains auditable.

## Distributions

Spatial distributions are area-weighted by latitude by default using `cos(latitude)`, which prevents latitude/longitude grid cells from being treated as equal-area cells over larger north-south extents.

Histogram comparison is the default:

```python
plot_kwargs={"distribution_kind": "hist"}
```

A weighted ECDF is also available:

```python
plot_kwargs={"distribution_kind": "ecdf"}
```

For a small compact study region, `area_weighted=False` can be used if a simple grid-cell distribution is desired.

When a decimated quick-look grid is used, the reported spatial distribution is naturally the distribution of the retained cells. Use native resolution when the distribution summary itself, rather than visualization speed, is the inferential target.

## Maps only or distributions only

```python
# Compact distribution comparison
ssf.plot_dynamic_conditions(
    era5,
    variables=["upward_potential", "sshf"],
    times=[...],
    plot_kwargs={"kind": "distribution"},
)

# Spatial fields only
ssf.plot_dynamic_conditions(
    era5,
    variables=["upward_potential", "sshf"],
    times=[...],
    plot_kwargs={"kind": "map"},
)
```

For each variable, maps across times use the same plotting limits. `robust=True` uses a common 1--99% range by default so isolated extreme cells do not flatten the visual contrast; change `quantile_range` or set `robust=False` when the extremes themselves are the feature of interest.

## Derived fields

The plotter can inspect a derived field without first adding it to the SSF choice table. A callable receives the bounded xarray Dataset at each selected time:

```python
conditions = ssf.plot_dynamic_conditions(
    era5,
    variables=["w", "sshf"],
    times=["2025-01-15 09:00", "2025-01-15 12:00"],
    timezone="Europe/Madrid",
    derived={
        "flight_condition_index": lambda ds: (
            ds["w"] + (-ds["sshf"] / 3600.0) / 100.0
        )
    },
)
```

The package does not assign a physical interpretation to such a derived index; the formula remains explicit in user code.

## Functional API

The same workflow is available without an SSF analysis object:

```python
from hsa.ssf import (
    extract_dynamic_condition_snapshots,
    plot_dynamic_condition_comparison,
    plot_dynamic_conditions,
    summarize_dynamic_conditions,
)
```

This separation is useful when the goal is to inspect an ERA5 field before constructing an SSF at all.
