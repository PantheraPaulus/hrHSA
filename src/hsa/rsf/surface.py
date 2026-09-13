from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler

from hsa.types import FeatureSpec


def split_multiscale_name(name: str) -> tuple[str, str | None]:
    """Split names such as ``ndvi_mean_90m`` into base variable and scale suffix."""

    base, sep, suffix = name.rpartition("_")
    if sep and suffix.endswith("m"):
        return base, suffix
    return name, None


def _get_template_layer(stack: xr.DataArray) -> xr.DataArray:
    """Return a single-band template layer from a ``band, y, x`` stack."""

    if "band" not in stack.dims:
        return stack
    return stack.sel(band=stack.band.values[0])


def _ensure_rio_crs(layer: xr.DataArray, *, name: str) -> None:
    """Raise a clear error if a layer has no rioxarray CRS."""

    try:
        crs = layer.rio.crs
    except Exception as exc:
        raise ValueError(
            f"Layer {name!r} has no rioxarray spatial metadata. "
            "Write a CRS with layer.rio.write_crs(...) before prediction."
        ) from exc
    if crs is None:
        raise ValueError(
            f"Layer {name!r} has no CRS. Write one with layer.rio.write_crs(...) before prediction."
        )


def resample_layer_to_template(
    layer: xr.DataArray,
    template: xr.DataArray,
    *,
    resampling: str = "nearest",
) -> xr.DataArray:
    """Reproject/resample one layer to match another layer's grid.

    Parameters
    ----------
    layer:
        Source layer to reproject.
    template:
        Target grid. Usually one band from the target-resolution stack.
    resampling:
        One of ``"nearest"``, ``"bilinear"``, ``"cubic"``, ``"average"``,
        ``"mode"``, ``"min"`` or ``"max"``.
    """

    from rasterio.enums import Resampling

    resampling_methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "min": Resampling.min,
        "max": Resampling.max,
    }
    if resampling not in resampling_methods:
        raise ValueError(
            f"Unsupported resampling method {resampling!r}. "
            f"Choose one of {sorted(resampling_methods)}."
        )

    _ensure_rio_crs(layer, name="source layer")
    _ensure_rio_crs(template, name="template layer")
    return layer.rio.reproject_match(template, resampling=resampling_methods[resampling])


def predict_rsf_surface(
    env: xr.DataArray,
    model,
    scaler: StandardScaler,
    spec: FeatureSpec,
    meta: dict[str, Any],
) -> xr.DataArray:
    """Project a fitted RSF model onto a raster stack with matching band names."""

    predictors = [*spec.linear, *spec.categorical]
    if not predictors:
        raise ValueError("FeatureSpec must contain at least one linear or categorical predictor.")

    coef = model.params
    eta = xr.zeros_like(env.sel(band=predictors[0])) + float(coef.get("const", 0.0))
    valid_mask = xr.ones_like(eta, dtype=bool)

    z: dict[str, xr.DataArray] = {}
    for index, variable in enumerate(spec.linear):
        layer = env.sel(band=variable)
        z[variable] = (layer - float(scaler.mean_[index])) / float(scaler.scale_[index])
        if variable in coef.index:
            eta = eta + z[variable] * float(coef[variable])

    for variable in spec.quadratic:
        name = f"{variable}__sq"
        if name in coef.index:
            eta = eta + (z[variable] ** 2) * float(coef[name])

    for left, right in spec.interactions:
        name = f"{left}__x__{right}"
        if name in coef.index:
            eta = eta + (z[left] * z[right]) * float(coef[name])

    for variable in spec.categorical:
        layer = env.sel(band=variable)
        info = meta["categorical"][variable]
        keep_levels = info["keep_levels"]
        valid_mask = valid_mask & layer.isin(keep_levels)

        for level in keep_levels:
            if level == info["reference"]:
                continue
            name = f"{variable}_{level}"
            if name in coef.index:
                eta = eta + xr.where(layer == level, float(coef[name]), 0.0)

    rsf = np.exp(eta.where(valid_mask)).rename("rsf").expand_dims(band=["rsf"])
    try:
        crs = env.rio.crs
        if crs is not None:
            rsf = rsf.rio.write_crs(crs)
    except Exception:
        pass
    return rsf


def build_prediction_stack(
    env_by_scale: dict[str, xr.DataArray],
    predictors: list[str],
    *,
    target_scale: str,
    categorical_predictors: list[str] | tuple[str, ...] | None = None,
    continuous_resampling: str = "bilinear",
    categorical_resampling: str = "nearest",
) -> xr.DataArray:
    """Build a common-grid raster stack matching fitted multiscale predictor names.

    ``env_by_scale`` should contain stacks keyed by suffixes such as ``"30m"``,
    ``"90m"`` and ``"300m"``. Predictors must use matching suffixes, for
    example ``"ndvi_mean_90m"``. Layers from non-target scales are reprojected
    and resampled to the target stack with ``rioxarray.reproject_match``.
    """

    if target_scale not in env_by_scale:
        raise KeyError(f"target_scale={target_scale!r} not found. Available: {list(env_by_scale)}")

    categorical_predictors = set(categorical_predictors or [])
    target_stack = env_by_scale[target_scale]
    template = _get_template_layer(target_stack)
    _ensure_rio_crs(template, name=f"target scale {target_scale!r}")

    layers: list[xr.DataArray] = []
    for predictor in predictors:
        base, scale = split_multiscale_name(predictor)
        if scale is None:
            raise ValueError(f"Predictor {predictor!r} has no scale suffix, e.g. '_30m'.")
        if scale not in env_by_scale:
            raise KeyError(f"Scale {scale!r} required by {predictor!r} not found in env_by_scale.")

        source_stack = env_by_scale[scale]
        if base not in source_stack.band.values:
            raise KeyError(
                f"Band {base!r} required by predictor {predictor!r} not found in scale {scale!r}."
            )

        source_layer = source_stack.sel(band=base)
        _ensure_rio_crs(source_layer, name=f"{base!r} at scale {scale!r}")

        if scale == target_scale:
            layer = source_layer
        else:
            method = categorical_resampling if predictor in categorical_predictors else continuous_resampling
            layer = resample_layer_to_template(source_layer, template, resampling=method)

        layer = layer.rename(predictor).expand_dims(band=[predictor])
        try:
            layer = layer.rio.write_crs(template.rio.crs)
        except Exception:
            pass
        layers.append(layer)

    stack = xr.concat(layers, dim="band").transpose("band", "y", "x")
    try:
        stack = stack.rio.write_crs(template.rio.crs)
    except Exception:
        pass
    return stack


def predict_rsf_surface_multiscale(
    env_by_scale: dict[str, xr.DataArray],
    *,
    target_scale: str,
    model,
    scaler: StandardScaler,
    spec: FeatureSpec,
    meta: dict[str, Any],
    continuous_resampling: str = "bilinear",
    categorical_resampling: str = "nearest",
) -> xr.DataArray:
    """Project a fitted multiscale RSF model onto raster stacks."""

    predictors = [*spec.linear, *spec.categorical]
    stack = build_prediction_stack(
        env_by_scale,
        predictors,
        target_scale=target_scale,
        categorical_predictors=spec.categorical,
        continuous_resampling=continuous_resampling,
        categorical_resampling=categorical_resampling,
    )
    return predict_rsf_surface(stack, model, scaler, spec, meta)
