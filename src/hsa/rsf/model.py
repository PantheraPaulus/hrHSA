from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from hsa.features import build_design_matrix
from hsa.types import FeatureSpec


@dataclass(frozen=True)
class PreparedRSFDesign:
    """Reusable frequentist RSF design matrix and fitting metadata.

    Preparing the model matrix can be a substantial fraction of total fitting time
    for large, low-dimensional RSFs. This container lets repeated optimizer/model
    fits reuse exactly the same ``X`` and ``y`` together with scaling/encoding
    metadata.
    """

    x: pd.DataFrame
    y: pd.Series
    scaler: StandardScaler
    spec: FeatureSpec
    meta: dict[str, Any]

    @property
    def n_rows(self) -> int:
        return int(len(self.x))

    @property
    def n_columns(self) -> int:
        return int(self.x.shape[1])


def prepare_rsf_design(
    df: pd.DataFrame,
    spec: FeatureSpec,
    *,
    min_available_proportion: float = 0.0,
    clean: bool = True,
) -> PreparedRSFDesign:
    """Clean samples and build a reusable RSF design matrix once."""

    if "used" not in df.columns:
        raise KeyError("prepare_rsf_design requires a boolean or 0/1 'used' column.")

    work = df.reset_index(drop=True).copy()
    if clean:
        candidate_cols = ["used", *spec.linear, *spec.categorical]
        candidate_cols = [col for col in candidate_cols if col in work.columns]
        work = (
            work.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=candidate_cols)
            .reset_index(drop=True)
        )

    x, scaler, meta = build_design_matrix(
        work,
        spec,
        scaler=None,
        fit_scaler=True,
        min_available_proportion=min_available_proportion,
        meta=None,
    )
    y = work.loc[x.index, "used"].astype(int)
    return PreparedRSFDesign(x=x, y=y, scaler=scaler, spec=spec, meta=meta)


def fit_prepared_rsf(
    prepared: PreparedRSFDesign,
    *,
    method: str = "newton",
    fit_kwargs: Mapping[str, Any] | None = None,
):
    """Fit Statsmodels Logit from an already prepared RSF design matrix."""

    options: dict[str, Any] = {"disp": False}
    if fit_kwargs is not None:
        options.update(dict(fit_kwargs))
    return sm.Logit(prepared.y, prepared.x).fit(method=method, **options)


def fit_rsf(
    df: pd.DataFrame,
    spec: FeatureSpec,
    *,
    min_available_proportion: float = 0.0,
    clean: bool = True,
    method: str = "newton",
    fit_kwargs: Mapping[str, Any] | None = None,
) -> tuple[sm.discrete.discrete_model.BinaryResultsWrapper, StandardScaler, FeatureSpec, dict[str, Any]]:
    """Fit a logistic RSF from used/available samples.

    ``method`` and ``fit_kwargs`` are forwarded to :meth:`statsmodels.Logit.fit`.
    The default remains Statsmodels' Newton optimizer for backwards-compatible
    scientific results. For very large, low-dimensional RSFs, ``method='lbfgs'``
    is the recommended performance candidate after verifying convergence and
    agreement with the reference fit.

    Use :func:`prepare_rsf_design` plus :func:`fit_prepared_rsf` when several fits
    should reuse exactly the same model matrix.
    """

    prepared = prepare_rsf_design(
        df,
        spec,
        min_available_proportion=min_available_proportion,
        clean=clean,
    )
    model = fit_prepared_rsf(prepared, method=method, fit_kwargs=fit_kwargs)
    return model, prepared.scaler, prepared.spec, prepared.meta


def predict_rsf_points(
    df: pd.DataFrame,
    model,
    scaler,
    spec: FeatureSpec,
    meta: dict[str, Any],
    *,
    pred_col: str = "rsf_pred",
) -> pd.DataFrame:
    """Predict relative selection scores for point samples."""

    df = df.reset_index(drop=True).copy()

    x, _, _ = build_design_matrix(
        df,
        spec,
        scaler=scaler,
        fit_scaler=False,
        meta=meta,
    )

    x = x[model.params.index]
    eta = model.predict(x, which="linear")

    out = df.loc[x.index].copy()
    out[pred_col] = np.exp(eta).to_numpy()

    return out
