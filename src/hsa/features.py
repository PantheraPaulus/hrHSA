from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from hsa.types import FeatureSpec


def _finalize_design_matrix(
    x: pd.DataFrame,
    spec: FeatureSpec,
    meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply row/column filtering and fitted column-order metadata."""
    x = x.dropna(axis=0)
    x = x.loc[:, x.nunique(dropna=False) > 1]

    if spec.add_const:
        x = sm.add_constant(x, has_constant="add")

    if meta["columns"] is None:
        meta["columns"] = x.columns.tolist()
    else:
        for column in meta["columns"]:
            if column not in x.columns:
                x[column] = 0.0
        x = x[meta["columns"]]

    return x, meta


def _build_continuous_design_matrix(
    df: pd.DataFrame,
    spec: FeatureSpec,
    scaler: StandardScaler | None,
    *,
    fit_scaler: bool,
    meta: dict[str, Any],
) -> tuple[pd.DataFrame, StandardScaler, dict[str, Any]]:
    """Fast path for continuous-only models.

    The general feature builder is intentionally pandas-oriented because it must
    handle categorical level bookkeeping. Large RSFs with continuous predictors
    do not need that machinery. Build their standardized/nonlinear columns in one
    preallocated NumPy matrix and create a DataFrame only once at the end.
    """
    names = list(spec.linear)
    raw = df.loc[:, names].to_numpy(dtype=float, copy=False)

    if scaler is None:
        scaler = StandardScaler()
    scaled = scaler.fit_transform(raw) if fit_scaler else scaler.transform(raw)
    scaled = np.asarray(scaled)

    linear_index = {name: index for index, name in enumerate(names)}
    for variable in spec.quadratic:
        if variable not in linear_index:
            raise KeyError(f"Quadratic term '{variable}' is not in spec.linear.")
    for left, right in spec.interactions:
        missing = [value for value in (left, right) if value not in linear_index]
        if missing:
            raise KeyError(
                f"Interaction {left!r} x {right!r} contains non-linear variables: {missing}"
            )

    derived_names = [f"{variable}__sq" for variable in spec.quadratic]
    derived_names.extend(
        f"{left}__x__{right}" for left, right in spec.interactions
    )
    columns = [*names, *derived_names]

    matrix = np.empty((len(df), len(columns)), dtype=scaled.dtype)
    matrix[:, : len(names)] = scaled
    cursor = len(names)

    for variable in spec.quadratic:
        source = scaled[:, linear_index[variable]]
        np.multiply(source, source, out=matrix[:, cursor])
        cursor += 1

    for left, right in spec.interactions:
        np.multiply(
            scaled[:, linear_index[left]],
            scaled[:, linear_index[right]],
            out=matrix[:, cursor],
        )
        cursor += 1

    x = pd.DataFrame(matrix, columns=columns, index=pd.RangeIndex(len(df)))
    x, meta = _finalize_design_matrix(x, spec, meta)
    return x, scaler, meta


def build_design_matrix(
    df: pd.DataFrame,
    spec: FeatureSpec,
    scaler: StandardScaler | None = None,
    *,
    fit_scaler: bool = False,
    min_available_proportion: float = 0.0,
    meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, StandardScaler, dict[str, Any]]:
    """Build a model matrix from a :class:`FeatureSpec`.

    The function is deliberately stateful via ``scaler`` and ``meta``. During
    fitting, call it with ``fit_scaler=True`` and ``meta=None``. During
    prediction, pass the fitted scaler and metadata so categorical levels and
    column order match the fitted model.

    Continuous-only specifications use a preallocated NumPy fast path to avoid
    repeated DataFrame copies and incremental column allocation on multi-million
    row RSFs. Models containing categorical predictors retain the reference
    pandas encoding path.
    """

    if meta is None:
        meta = {"categorical": {}, "columns": None}

    if spec.linear and not spec.categorical:
        return _build_continuous_design_matrix(
            df,
            spec,
            scaler,
            fit_scaler=fit_scaler,
            meta=meta,
        )

    df = df.reset_index(drop=True).copy()
    parts: list[pd.DataFrame] = []

    if spec.linear:
        x_cont = df[spec.linear].copy()
        if scaler is None:
            scaler = StandardScaler()
        arr = scaler.fit_transform(x_cont) if fit_scaler else scaler.transform(x_cont)
        x_scaled = pd.DataFrame(arr, columns=spec.linear, index=df.index)

        for variable in spec.quadratic:
            if variable not in x_scaled.columns:
                raise KeyError(f"Quadratic term '{variable}' is not in spec.linear.")
            x_scaled[f"{variable}__sq"] = x_scaled[variable] ** 2

        for left, right in spec.interactions:
            missing = [v for v in (left, right) if v not in x_scaled.columns]
            if missing:
                raise KeyError(
                    f"Interaction {left!r} x {right!r} contains non-linear variables: {missing}"
                )
            x_scaled[f"{left}__x__{right}"] = x_scaled[left] * x_scaled[right]

        parts.append(x_scaled)
    else:
        if scaler is None:
            scaler = StandardScaler()

    if spec.categorical:
        for variable in spec.categorical:
            if variable not in meta["categorical"]:
                if not fit_scaler:
                    raise ValueError(
                        f"No categorical metadata found for '{variable}'. "
                        "Pass metadata from fitting when predicting."
                    )
                if "used" not in df.columns:
                    raise KeyError("Categorical encoding during fitting requires a 'used' column.")

                available = df.loc[df["used"] != True]
                used = df.loc[df["used"] == True]

                available_props = available[variable].value_counts(normalize=True, dropna=True)
                keep_levels = available_props[available_props >= min_available_proportion].index.tolist()

                used_counts = used[variable].value_counts(dropna=True)
                keep_levels = [level for level in keep_levels if used_counts.get(level, 0) > 0]

                if not keep_levels:
                    raise ValueError(
                        f"No levels of '{variable}' remain after requiring minimum availability "
                        f"proportion {min_available_proportion} and at least one used point."
                    )

                reference = available_props.loc[keep_levels].idxmax()
                ordered_levels = [reference] + [level for level in keep_levels if level != reference]
                meta["categorical"][variable] = {
                    "keep_levels": keep_levels,
                    "reference": reference,
                    "ordered_levels": ordered_levels,
                }

            info = meta["categorical"][variable]
            series = df[variable].where(df[variable].isin(info["keep_levels"]), np.nan)
            series = pd.Categorical(series, categories=info["ordered_levels"])
            dummies = pd.get_dummies(series, prefix=variable, dtype=float, dummy_na=False)

            reference_column = f"{variable}_{info['reference']}"
            if reference_column in dummies.columns:
                dummies = dummies.drop(columns=reference_column)

            dummies.index = df.index
            parts.append(dummies)

    x = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    x, meta = _finalize_design_matrix(x, spec, meta)
    return x, scaler, meta
