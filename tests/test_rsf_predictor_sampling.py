import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point, box

import hsa.rsf.base as base_module
from hsa.rsf.base import RSFAnalysis, RSFFit


class _DummyFit(RSFFit):
    def summary(self, *args, **kwargs):
        return None

    def coefficients(self, *args, **kwargs):
        return pd.DataFrame()

    def predict_surface(self, env=None, *args, **kwargs):
        return None


class _DummyAnalysis(RSFAnalysis):
    def fit(self, *args, **kwargs):
        return _DummyFit()

    def _run_loio(self, scheme, **kwargs):
        return None


def test_shared_analysis_samples_only_fitted_predictors(monkeypatch):
    predictors = ["elevation", "slope"]
    reloc = gpd.GeoDataFrame(
        {
            "animal": ["bird-A", "bird-A"],
            "Timestamp": pd.date_range(
                "2024-01-01",
                periods=2,
                freq="D",
                tz="UTC",
            ),
        },
        geometry=[
            Point(0.0, 0.0),
            Point(1.0, 1.0),
        ],
        crs="EPSG:3857",
    )
    domain = gpd.GeoDataFrame(
        geometry=[box(-1.0, -1.0, 2.0, 2.0)],
        crs=reloc.crs,
    )
    env = xr.DataArray(
        np.zeros((3, 2, 2), dtype=np.float32),
        dims=("band", "y", "x"),
        coords={
            "band": ["elevation", "slope", "unused"],
            "y": [0.0, 1.0],
            "x": [0.0, 1.0],
        },
    )

    analysis = _DummyAnalysis(
        reloc,
        env,
        predictors=predictors,
        id_col="animal",
        domain=domain,
    )
    analysis._domains = {"bird-A": domain}

    def fake_available_points(domain_i, n, *, used, **kwargs):
        out = used[["Timestamp", "geometry"]].copy()
        out["used"] = True
        return gpd.GeoDataFrame(
            out,
            geometry="geometry",
            crs=used.crs,
        )

    monkeypatch.setattr(
        base_module,
        "sample_available_points",
        fake_available_points,
    )

    captured = {}

    def fake_raster_stack(samples, env_i, *, bands=None, id_cols=None, **kwargs):
        captured["bands"] = list(bands)
        captured["source_bands"] = list(env_i["band"].values)
        out = pd.DataFrame(
            {
                "elevation": np.ones(len(samples)),
                "slope": np.ones(len(samples)) * 2,
                "used": samples["used"].to_numpy(),
                "Timestamp": samples["Timestamp"].to_numpy(),
            }
        )
        if id_cols is not None:
            out[id_cols] = samples[id_cols].to_numpy()
        env_used = env_i.sel(band=bands)
        return out, env_used

    monkeypatch.setattr(
        base_module,
        "sample_raster_stack",
        fake_raster_stack,
    )

    sampled = analysis.sample_training_data(
        sampling_factor=1,
        id_cols="animal",
    )

    assert captured["bands"] == predictors
    assert captured["source_bands"] == ["elevation", "slope", "unused"]
    assert set(predictors).issubset(sampled.columns)
    assert "unused" not in sampled.columns
