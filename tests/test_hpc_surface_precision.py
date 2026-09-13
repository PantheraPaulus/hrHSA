from types import SimpleNamespace

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers .rio
import xarray as xr

from hsa import FeatureSpec
from hsa.rsf import predict_rsf_surface, predict_rsf_surface_chunked


def _environment():
    x = np.arange(8, dtype=float)
    y = np.arange(7, -1, -1, dtype=float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    values = np.stack(
        [
            0.25 * xx + yy,
            np.sin(xx) - 0.1 * yy,
            np.cos(yy) + 0.2 * xx,
        ]
    ).astype("float32")
    return xr.DataArray(
        values,
        dims=("band", "y", "x"),
        coords={"band": ["a", "b", "c"], "y": y, "x": x},
    ).rio.write_crs("EPSG:3857")


def _model():
    spec = FeatureSpec(
        linear=["a", "b", "c"],
        quadratic=["a"],
        interactions=[("a", "b")],
        add_const=True,
    )
    model = SimpleNamespace(
        params=pd.Series(
            {
                "const": -0.3,
                "a": 0.4,
                "b": -0.2,
                "c": 0.1,
                "a__sq": 0.03,
                "a__x__b": -0.04,
            }
        )
    )
    scaler = SimpleNamespace(
        mean_=np.array([4.0, -0.5, 1.0]),
        scale_=np.array([2.5, 1.5, 2.0]),
    )
    meta = {"categorical": {}, "columns": list(model.params.index)}
    return model, scaler, spec, meta


def test_float32_fast_surface_matches_reference_and_preserves_output_dtype():
    env = _environment()
    model, scaler, spec, meta = _model()
    reference = predict_rsf_surface(env, model, scaler, spec, meta)
    accelerated = predict_rsf_surface_chunked(
        env,
        model,
        scaler,
        spec,
        meta,
        chunks={"band": -1, "y": 4, "x": 4},
        dtype="float32",
    )

    assert accelerated.dtype == np.dtype("float32")
    np.testing.assert_allclose(reference.values, accelerated.values, rtol=2e-5, atol=2e-6)


def test_float64_compute_mode_matches_float64_reference_tightly():
    env = _environment()
    model, scaler, spec, meta = _model()
    reference = predict_rsf_surface(env.astype("float64"), model, scaler, spec, meta)
    accelerated = predict_rsf_surface_chunked(
        env,
        model,
        scaler,
        spec,
        meta,
        chunks={"band": -1, "y": 4, "x": 4},
        dtype="float64",
        compute_dtype="float64",
    )

    assert accelerated.dtype == np.dtype("float64")
    np.testing.assert_allclose(reference.values, accelerated.values, rtol=1e-10, atol=1e-10)
