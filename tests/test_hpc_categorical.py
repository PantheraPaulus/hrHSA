from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers the xarray .rio accessor
import xarray as xr
from sklearn.preprocessing import StandardScaler

from hsa import FeatureSpec
from hsa.rsf import predict_rsf_surface, predict_rsf_surface_chunked


def test_categorical_only_fused_surface_matches_reference():
    env = xr.DataArray(
        np.array(
            [
                [
                    [0, 1, 2, 3],
                    [2, 1, 0, 1],
                    [1, 2, 2, 0],
                ]
            ],
            dtype="int16",
        ),
        dims=("band", "y", "x"),
        coords={
            "band": ["landcover"],
            "y": [60.0, 30.0, 0.0],
            "x": [0.0, 30.0, 60.0, 90.0],
        },
    ).rio.write_crs("EPSG:3857")

    spec = FeatureSpec(
        linear=[],
        categorical=["landcover"],
        add_const=True,
    )
    model = SimpleNamespace(
        params=pd.Series(
            {
                "const": -0.5,
                "landcover_1": 0.8,
                "landcover_2": -0.3,
            }
        )
    )
    scaler = StandardScaler()  # intentionally unfitted for categorical-only models
    meta = {
        "categorical": {
            "landcover": {
                "keep_levels": [0, 1, 2],
                "reference": 0,
                "ordered_levels": [0, 1, 2],
            }
        },
        "columns": ["const", "landcover_1", "landcover_2"],
    }

    reference = predict_rsf_surface(env, model, scaler, spec, meta)
    accelerated = predict_rsf_surface_chunked(
        env,
        model,
        scaler,
        spec,
        meta,
        chunks={"band": -1, "y": 2, "x": 2},
    )

    np.testing.assert_allclose(
        reference.values,
        accelerated.values,
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )
