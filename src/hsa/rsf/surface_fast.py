"""Fused chunk-wise RSF surface prediction.

The reference implementation in :mod:`hsa.rsf.surface` deliberately expresses
model terms as labelled xarray operations. This module provides an equivalent
HPC-oriented path that evaluates a complete model inside one spatial block,
reducing intermediate arrays and memory traffic.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hsa.compute.planning import plan_raster_chunks
from hsa.types import FeatureSpec


class _BlockWorkspace:
    """Reusable per-thread temporaries for one block shape/model structure.

    The final ``eta`` array is deliberately *not* stored here. It becomes the
    returned output block, so keeping it task-local avoids aliasing when the same
    worker thread immediately starts another Dask task. Only true temporaries are
    reused.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: np.dtype,
        retained: tuple[str, ...],
        *,
        has_linear_temp: bool,
        has_categorical: bool,
    ):
        self.scratch = np.empty(shape, dtype=dtype) if retained else None
        self.ztemp = np.empty(shape, dtype=dtype) if has_linear_temp else None
        self.standardized = {
            variable: np.empty(shape, dtype=dtype) for variable in retained
        }
        self.valid = np.empty(shape, dtype=bool) if has_categorical else None
        self.variable_valid = (
            np.empty(shape, dtype=bool) if has_categorical else None
        )
        self.level_mask = np.empty(shape, dtype=bool) if has_categorical else None


class _ThreadLocalWorkspacePool:
    """Small thread-local LRU of block workspaces.

    Dask worker processes commonly execute several blocks sequentially on the same
    thread. Reusing standardisation/scratch buffers removes repeated large NumPy
    allocations while preserving thread safety. The deliberately small LRU keeps
    the regular block and at most one edge-block shape without allowing long-lived
    workers to retain many model-sized arrays.
    """

    def __init__(self, max_entries: int = 2):
        self._local = threading.local()
        self.max_entries = int(max_entries)

    def get(
        self,
        shape: tuple[int, int],
        dtype: np.dtype,
        retained: tuple[str, ...],
        *,
        has_linear_temp: bool,
        has_categorical: bool,
    ) -> _BlockWorkspace:
        cache = getattr(self._local, "cache", None)
        if cache is None:
            cache = OrderedDict()
            self._local.cache = cache

        key = (
            shape,
            np.dtype(dtype).str,
            retained,
            bool(has_linear_temp),
            bool(has_categorical),
        )
        workspace = cache.pop(key, None)
        if workspace is None:
            workspace = _BlockWorkspace(
                shape,
                np.dtype(dtype),
                retained,
                has_linear_temp=has_linear_temp,
                has_categorical=has_categorical,
            )
        cache[key] = workspace

        while len(cache) > self.max_entries:
            cache.popitem(last=False)
        return workspace

    def clear_current_thread(self) -> None:
        """Release cached arrays owned by the calling thread."""
        self._local.cache = OrderedDict()


_BLOCK_WORKSPACES = _ThreadLocalWorkspacePool()


def _block_predict(
    block: xr.DataArray,
    *,
    coefficients: dict[str, float],
    means: list[float],
    scales: list[float],
    spec: FeatureSpec,
    meta: dict[str, Any],
    dtype: str,
    compute_dtype: str,
    reuse_workspace: bool = True,
) -> xr.DataArray:
    """Evaluate one complete spatial block using low-allocation NumPy arithmetic."""
    block = block.transpose("band", "y", "x")
    band_names = [str(value) for value in block["band"].values]
    band_index = {name: index for index, name in enumerate(band_names)}
    data = np.asarray(block.values)
    shape = (int(block.sizes["y"]), int(block.sizes["x"]))

    compute = np.dtype(compute_dtype)
    output = np.dtype(dtype)
    if compute.kind != "f" or output.kind != "f":
        raise TypeError("dtype and compute_dtype must both be floating-point dtypes.")
    scalar = compute.type

    # Retain standardised arrays only for nonlinear terms that actually survived
    # into the fitted coefficient vector. Screened/dropped nonlinear terms should
    # not force full-block standardisation buffers or memory passes.
    active_quadratic: list[tuple[str, str]] = []
    active_interactions: list[tuple[str, str, str]] = []
    retain_standardized: set[str] = set()
    for variable in spec.quadratic:
        name = f"{variable}__sq"
        if name in coefficients:
            active_quadratic.append((variable, name))
            retain_standardized.add(variable)
    for left, right in spec.interactions:
        name = f"{left}__x__{right}"
        if name in coefficients:
            active_interactions.append((left, right, name))
            retain_standardized.add(left)
            retain_standardized.add(right)

    retained_names = tuple(
        variable for variable in spec.linear if variable in retain_standardized
    )

    # Purely linear terms do not need a standardised raster at all. Algebraically
    #
    #   beta * (x - mean) / scale
    #       = (beta / scale) * x - beta * mean / scale
    #
    # so fold the constant offset into eta's intercept once and reduce each such
    # variable from four full-array passes (subtract/divide/multiply/add) to two
    # (multiply/add). The workstation PMU diagnostics identified memory-hierarchy
    # service latency as the dominant threaded bottleneck, making avoided passes
    # more valuable than avoiding a few scalar Python operations.
    adjusted_intercept = scalar(coefficients.get("const", 0.0))
    affine_slopes: dict[str, np.generic] = {}
    for index, variable in enumerate(spec.linear):
        scale = scalar(scales[index])
        if scale == 0:
            raise ValueError(f"Fitted scaler has zero scale for {variable!r}.")
        coefficient = coefficients.get(variable)
        if coefficient is None or variable in retain_standardized:
            continue
        coeff = scalar(coefficient)
        slope = scalar(coeff / scale)
        affine_slopes[variable] = slope
        adjusted_intercept = scalar(
            adjusted_intercept - scalar(means[index]) * slope
        )

    eta = np.full(shape, adjusted_intercept, dtype=compute)

    workspace = None
    if reuse_workspace:
        workspace = _BLOCK_WORKSPACES.get(
            shape,
            compute,
            retained_names,
            has_linear_temp=bool(affine_slopes),
            has_categorical=bool(spec.categorical),
        )

    standardized: dict[str, np.ndarray] = {}
    scratch: np.ndarray | None = None
    ztemp: np.ndarray | None = None

    def get_scratch() -> np.ndarray:
        nonlocal scratch
        if workspace is not None:
            if workspace.scratch is None:
                raise RuntimeError("Internal workspace is missing a scratch buffer.")
            return workspace.scratch
        if scratch is None:
            scratch = np.empty(shape, dtype=compute)
        return scratch

    def get_linear_temp() -> np.ndarray:
        nonlocal ztemp
        if workspace is not None:
            if workspace.ztemp is None:
                raise RuntimeError("Internal workspace is missing a linear buffer.")
            return workspace.ztemp
        if ztemp is None:
            ztemp = np.empty(shape, dtype=compute)
        return ztemp

    for index, variable in enumerate(spec.linear):
        if variable not in band_index:
            raise KeyError(f"Predictor {variable!r} is missing from prediction block.")

        coefficient = coefficients.get(variable)
        needs_later = variable in retain_standardized
        if coefficient is None and not needs_later:
            continue

        layer = np.asarray(data[band_index[variable]], dtype=compute)

        if not needs_later:
            # The scaler offset was folded into adjusted_intercept above, leaving
            # only slope*x and eta+=term. This is the common path for linear-only
            # predictors in broad environmental stacks.
            z = get_linear_temp()
            np.multiply(layer, affine_slopes[variable], out=z)
            np.add(eta, z, out=eta)
            continue

        if workspace is None:
            z = np.empty(shape, dtype=compute)
        else:
            z = workspace.standardized[variable]

        scale = scalar(scales[index])
        np.subtract(layer, scalar(means[index]), out=z)
        np.divide(z, scale, out=z)

        if coefficient is not None:
            work = get_scratch()
            np.multiply(z, scalar(coefficient), out=work)
            np.add(eta, work, out=eta)

        standardized[variable] = z

    for variable, name in active_quadratic:
        work = get_scratch()
        z = standardized[variable]
        np.multiply(z, z, out=work)
        np.multiply(work, scalar(coefficients[name]), out=work)
        np.add(eta, work, out=eta)

    for left, right, name in active_interactions:
        work = get_scratch()
        np.multiply(standardized[left], standardized[right], out=work)
        np.multiply(work, scalar(coefficients[name]), out=work)
        np.add(eta, work, out=eta)

    valid: np.ndarray | None = None
    if workspace is not None and spec.categorical:
        if (
            workspace.valid is None
            or workspace.variable_valid is None
            or workspace.level_mask is None
        ):
            raise RuntimeError("Internal workspace is missing categorical buffers.")
        valid = workspace.valid
        valid.fill(True)

    for variable in spec.categorical:
        if variable not in band_index:
            raise KeyError(
                f"Categorical predictor {variable!r} is missing from prediction block."
            )
        layer = data[band_index[variable]]
        info = meta["categorical"][variable]
        keep_levels = list(info["keep_levels"])

        if workspace is None:
            variable_valid = np.isin(layer, keep_levels)
            if valid is None:
                valid = variable_valid
            else:
                np.logical_and(valid, variable_valid, out=valid)

            for level in keep_levels:
                if level == info["reference"]:
                    continue
                name = f"{variable}_{level}"
                if name in coefficients:
                    level_mask = layer == level
                    eta[level_mask] += scalar(coefficients[name])
        else:
            variable_valid = workspace.variable_valid
            level_mask = workspace.level_mask
            assert variable_valid is not None and level_mask is not None
            variable_valid.fill(False)

            # Build the valid-level mask and categorical contribution with two
            # reusable boolean arrays instead of allocating np.isin/layer==level
            # temporaries for every task.
            for level in keep_levels:
                np.equal(layer, level, out=level_mask)
                np.logical_or(variable_valid, level_mask, out=variable_valid)
                if level == info["reference"]:
                    continue
                name = f"{variable}_{level}"
                if name in coefficients:
                    eta[level_mask] += scalar(coefficients[name])

            assert valid is not None
            np.logical_and(valid, variable_valid, out=valid)

    if valid is not None:
        if workspace is not None and workspace.level_mask is not None:
            np.logical_not(valid, out=workspace.level_mask)
            eta[workspace.level_mask] = np.nan
        else:
            eta[~valid] = np.nan

    # Evaluate exp in place. The output block remains task-owned; only scratch
    # arrays are thread-local. This avoids both another full-size exp result and
    # unsafe output aliasing across sequential Dask tasks.
    np.exp(eta, out=eta)
    out = eta.astype(output, copy=False)
    return xr.DataArray(
        out,
        dims=("y", "x"),
        coords={"y": block["y"], "x": block["x"]},
        name="rsf",
    )


def predict_rsf_surface_chunked(
    env: xr.DataArray,
    model,
    scaler,
    spec: FeatureSpec,
    meta: dict[str, Any],
    *,
    chunks: dict[str, int] | None = None,
    target_chunk_mb: int = 256,
    align_storage: bool = True,
    dtype: str = "float32",
    compute_dtype: str | None = None,
    reuse_workspace: bool = True,
) -> xr.DataArray:
    """Project an RSF by fusing all model arithmetic within each raster block.

    The returned object remains lazy when ``env`` is Dask-backed. The complete
    set of required bands is kept in one band chunk while x/y are partitioned
    according to the requested memory target and, by default, aligned to source
    storage chunks.

    ``compute_dtype`` controls intermediate arithmetic independently of the output
    dtype. By default it follows ``dtype``; therefore the normal accelerated
    ``dtype='float32'`` path no longer silently promotes every raster layer and
    intermediate to float64. Use ``compute_dtype='float64'`` when a benchmark or
    analysis explicitly requires double-precision intermediate arithmetic.

    ``reuse_workspace=True`` reuses per-thread standardisation, nonlinear scratch,
    and categorical mask buffers across blocks. The final output buffer is never
    reused, so lazy/distributed results cannot alias a later task. Set it to
    ``False`` for diagnostic comparisons or unusually memory-constrained workers.
    """
    predictors = list(dict.fromkeys([*spec.linear, *spec.categorical]))
    if not predictors:
        raise ValueError("FeatureSpec must contain at least one predictor.")
    if "band" not in env.dims:
        raise ValueError("env must contain a 'band' dimension.")

    output_dtype = np.dtype(dtype)
    effective_compute_dtype = np.dtype(dtype if compute_dtype is None else compute_dtype)
    if output_dtype.kind != "f" or effective_compute_dtype.kind != "f":
        raise TypeError("dtype and compute_dtype must both be floating-point dtypes.")

    available = {str(value) for value in env["band"].values}
    missing = [value for value in predictors if value not in available]
    if missing:
        raise KeyError(f"Prediction bands not found in env: {missing}")

    raster = env.sel(band=predictors).transpose("band", "y", "x")
    if chunks is None:
        chunks = plan_raster_chunks(
            raster,
            workload="surface_prediction",
            target_chunk_mb=target_chunk_mb,
            align_storage=align_storage,
            selected_band_count=len(predictors),
        )

    if getattr(raster.data, "chunks", None) is not None:
        raster = raster.chunk({**chunks, "band": -1})

    params = pd.Series(model.params)
    coefficients = {str(name): float(value) for name, value in params.items()}
    if spec.linear:
        means = [float(value) for value in scaler.mean_]
        scales = [float(value) for value in scaler.scale_]
    else:
        # build_design_matrix intentionally leaves the StandardScaler unfitted for
        # categorical-only models. The reference raster predictor never touches it,
        # so the fused path must likewise avoid requiring mean_/scale_.
        means = []
        scales = []
    kwargs = {
        "coefficients": coefficients,
        "means": means,
        "scales": scales,
        "spec": spec,
        "meta": meta,
        "dtype": output_dtype.name,
        "compute_dtype": effective_compute_dtype.name,
        "reuse_workspace": bool(reuse_workspace),
    }

    if getattr(raster.data, "chunks", None) is None:
        predicted = _block_predict(raster, **kwargs)
    else:
        # map_blocks requires each returned block to expose exactly the coordinates
        # declared by the template. Drop scalar ``band``/``spatial_ref`` metadata;
        # CRS metadata is restored once the blockwise graph has been assembled.
        template = (
            xr.zeros_like(
                raster.isel(band=0, drop=True),
                dtype=output_dtype,
            )
            .reset_coords(drop=True)
            .rename("rsf")
        )
        predicted = xr.map_blocks(
            _block_predict,
            raster,
            kwargs=kwargs,
            template=template,
        )

    predicted = predicted.expand_dims(band=["rsf"]).transpose("band", "y", "x")
    try:
        crs = env.rio.crs
        if crs is not None:
            predicted = predicted.rio.write_crs(crs)
    except Exception:
        pass
    return predicted
