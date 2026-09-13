from __future__ import annotations

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 - registers the xarray .rio accessor
import xarray as xr

from hsa import FeatureSpec
from hsa.compute import plan_raster_chunks, sample_raster_stack_chunked
from hsa.rsf import predict_rsf_surface, predict_rsf_surface_chunked
from hsa.sampling import sample_raster_stack


def _dask_env() -> xr.DataArray:
    pytest.importorskip("dask.array")
    x = np.arange(0.0, 120.0, 10.0)
    y = np.arange(110.0, -10.0, -10.0)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    env = xr.DataArray(
        np.stack(
            [
                xx + 0.5 * yy,
                0.05 * xx**2 - 0.1 * yy,
            ]
        ).astype("float32"),
        dims=("band", "y", "x"),
        coords={"band": ["a", "b"], "y": y, "x": x},
    ).rio.write_crs("EPSG:3857")
    return env.chunk({"band": -1, "y": 4, "x": 3})


def _samples() -> gpd.GeoDataFrame:
    x = np.array([3.0, 24.0, 59.0, 87.0, 112.0])
    y = np.array([108.0, 72.0, 43.0, 15.0, 2.0])
    return gpd.GeoDataFrame(
        {"used": [True, False, True, False, True]},
        geometry=gpd.points_from_xy(x, y),
        crs="EPSG:3857",
    )


def test_storage_aligned_planner_uses_dask_source_chunks():
    env = _dask_env()
    chunks = plan_raster_chunks(
        env,
        workload="point_sampling",
        target_chunk_mb=1,
        align_storage=True,
    )
    assert chunks["band"] == -1
    assert chunks["y"] % 4 == 0 or chunks["y"] == env.sizes["y"]
    assert chunks["x"] % 3 == 0 or chunks["x"] == env.sizes["x"]


def test_dask_chunked_sampler_matches_reference():
    env = _dask_env()
    samples = _samples()

    reference = sample_raster_stack(samples, env, bands=["a", "b"])
    accelerated = sample_raster_stack_chunked(
        samples,
        env,
        bands=["a", "b"],
        chunks={"band": -1, "y": 4, "x": 3},
    )
    np.testing.assert_allclose(
        reference[["a", "b"]].to_numpy(),
        accelerated[["a", "b"]].to_numpy(),
        rtol=0,
        atol=1e-6,
    )


def test_distributed_chunked_sampler_matches_reference():
    distributed = pytest.importorskip("distributed")
    env = _dask_env()
    samples = _samples()
    reference = sample_raster_stack(samples, env, bands=["a", "b"])

    cluster = distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    try:
        accelerated = sample_raster_stack_chunked(
            samples,
            env,
            bands=["a", "b"],
            chunks={"band": -1, "y": 4, "x": 3},
            client=client,
        )
    finally:
        client.close()
        cluster.close()

    np.testing.assert_allclose(
        reference[["a", "b"]].to_numpy(),
        accelerated[["a", "b"]].to_numpy(),
        rtol=0,
        atol=1e-6,
    )


def test_dask_fused_surface_matches_reference():
    env = _dask_env()
    spec = FeatureSpec(
        linear=["a", "b"],
        quadratic=["a"],
        interactions=[("a", "b")],
        add_const=True,
    )
    model = SimpleNamespace(
        params=pd.Series(
            {
                "const": -0.4,
                "a": 0.3,
                "b": -0.2,
                "a__sq": 0.02,
                "a__x__b": 0.01,
            }
        )
    )
    scaler = SimpleNamespace(
        mean_=np.array([80.0, 180.0]),
        scale_=np.array([40.0, 150.0]),
    )
    meta = {"categorical": {}, "columns": list(model.params.index)}

    reference = predict_rsf_surface(env, model, scaler, spec, meta).compute()
    accelerated = predict_rsf_surface_chunked(
        env,
        model,
        scaler,
        spec,
        meta,
        chunks={"band": -1, "y": 4, "x": 3},
    ).compute()
    np.testing.assert_allclose(
        reference.values,
        accelerated.values,
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
    )
