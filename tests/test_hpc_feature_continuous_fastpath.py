from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hsa.features import build_design_matrix
from hsa.types import FeatureSpec


def test_continuous_fastpath_matches_manual_design_matrix():
    df = pd.DataFrame(
        {
            "x0": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x1": [4.0, 3.0, 2.0, 1.0, 0.0],
            "used": [0, 0, 1, 0, 1],
        }
    )
    spec = FeatureSpec(
        linear=["x0", "x1"],
        quadratic=["x0"],
        interactions=[("x0", "x1")],
        add_const=True,
    )

    x, scaler, meta = build_design_matrix(df, spec, fit_scaler=True)

    scaled = StandardScaler().fit_transform(df[["x0", "x1"]])
    expected = pd.DataFrame(
        {
            "const": 1.0,
            "x0": scaled[:, 0],
            "x1": scaled[:, 1],
            "x0__sq": scaled[:, 0] ** 2,
            "x0__x__x1": scaled[:, 0] * scaled[:, 1],
        }
    )

    pd.testing.assert_frame_equal(x, expected)
    np.testing.assert_allclose(scaler.mean_, [2.0, 2.0])
    assert meta["columns"] == expected.columns.tolist()


def test_continuous_fastpath_reuses_scaler_and_column_order():
    train = pd.DataFrame(
        {
            "x0": [0.0, 1.0, 2.0, 3.0],
            "x1": [1.0, 3.0, 2.0, 4.0],
            "used": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "x0": [1.5, 2.5],
            "x1": [2.5, 3.5],
            "used": [1, 0],
        },
        index=[20, 21],
    )
    spec = FeatureSpec(
        linear=["x0", "x1"],
        quadratic=["x0"],
        interactions=[("x0", "x1")],
        add_const=True,
    )

    _, scaler, meta = build_design_matrix(train, spec, fit_scaler=True)
    predicted, _, returned_meta = build_design_matrix(
        test,
        spec,
        scaler=scaler,
        fit_scaler=False,
        meta=meta,
    )

    scaled = scaler.transform(test[["x0", "x1"]])
    np.testing.assert_allclose(predicted["x0"], scaled[:, 0])
    np.testing.assert_allclose(predicted["x1"], scaled[:, 1])
    np.testing.assert_allclose(predicted["x0__sq"], scaled[:, 0] ** 2)
    np.testing.assert_allclose(
        predicted["x0__x__x1"],
        scaled[:, 0] * scaled[:, 1],
    )
    assert predicted.index.tolist() == [0, 1]
    assert predicted.columns.tolist() == meta["columns"]
    assert returned_meta is meta


def test_categorical_spec_keeps_reference_path():
    df = pd.DataFrame(
        {
            "x0": [0.0, 1.0, 2.0, 3.0],
            "habitat": ["grass", "grass", "wood", "wood"],
            "used": [0, 1, 0, 1],
        }
    )
    spec = FeatureSpec(
        linear=["x0"],
        categorical=["habitat"],
        add_const=True,
    )

    x, _, meta = build_design_matrix(df, spec, fit_scaler=True)

    assert "const" in x.columns
    assert "x0" in x.columns
    assert meta["categorical"]["habitat"]["reference"] in {"grass", "wood"}
    assert meta["columns"] == x.columns.tolist()
