from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

from hsa._time import require_timezone_aware

from hsa.types import FoldSplit, FeatureSpec
from hsa.sampling import get_availability_domain, sample_available_points, sample_raster_stack
from hsa.rsf.model import fit_rsf, predict_rsf_points
from hsa.rsf.surface import predict_rsf_surface

from hsa.rsf.validation import (
    boyce_quantile_bins,
    boyce_sliding_window,
    calibration_rsf_quantile_bins,
)

def assign_week_folds(reloc: gpd.GeoDataFrame, *, k: int = 5, seed: int = 42) -> gpd.GeoDataFrame:
    """Assign whole ISO weeks to cross-validation folds."""

    g = reloc.copy()
    iso = g["Timestamp"].dt.isocalendar()
    g["year_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    weeks = g["year_week"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(weeks)
    g["fold"] = g["year_week"].map({week: i % k for i, week in enumerate(weeks)}).astype(int)
    return g


def thin_by_time(reloc: gpd.GeoDataFrame, *, min_dt: str = "12h") -> gpd.GeoDataFrame:
    """Thin records so retained fixes are at least ``min_dt`` apart."""

    g = reloc.sort_values("Timestamp").reset_index(drop=True).copy()
    min_delta = pd.Timedelta(min_dt)
    keep = np.zeros(len(g), dtype=bool)
    last_time = None
    for i, timestamp in enumerate(g["Timestamp"]):
        if last_time is None or pd.Timestamp(timestamp) - last_time >= min_delta:
            keep[i] = True
            last_time = pd.Timestamp(timestamp)
    return g.loc[keep].reset_index(drop=True)


def get_fold_split(
    reloc_folds: gpd.GeoDataFrame,
    fold_id: int,
    *,
    thin_train_dt: str = "12h",
    thin_test_dt: Optional[str] = None,
) -> FoldSplit:
    """Return train/test split for one fold, with optional temporal thinning."""

    train = reloc_folds.loc[reloc_folds["fold"] != fold_id].copy()
    test = reloc_folds.loc[reloc_folds["fold"] == fold_id].copy()
    train_thin = thin_by_time(train, min_dt=thin_train_dt)
    test_thin = test if thin_test_dt is None else thin_by_time(test, min_dt=thin_test_dt)
    return FoldSplit(train=train, test=test, train_thin=train_thin, test_thin=test_thin)


def cross_validate_rsf_temporal(
    reloc: gpd.GeoDataFrame,
    env,
    spec: FeatureSpec,
    *,
    k_folds: int = 5,
    sampling_factor_train: int = 10,
    n_background_boyce: int = 100_000,
    n_bins: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Temporal/week-blocked RSF validation.

    This is intentionally conservative and should later be complemented by
    leave-one-individual-out validation for multi-animal studies.
    """

    domain = get_availability_domain(reloc)
    reloc_folds = assign_week_folds(reloc, k=k_folds, seed=seed)
    rows = []

    for fold in range(k_folds):
        split = get_fold_split(reloc_folds, fold)
        train_samples = sample_available_points(
            domain,
            len(split.train_thin) * sampling_factor_train,
            used=split.train_thin,
            seed=seed + fold,
        )
        train_df = sample_raster_stack(train_samples, env)
        model, scaler, fitted_spec, meta = fit_rsf(train_df, spec)
        rsf = predict_rsf_surface(env, model, scaler, fitted_spec, meta)

        test = split.test.copy()
        test["used"] = True
        test_df = sample_raster_stack(test, env).replace([np.inf, -np.inf], np.nan)
        test_pred = predict_rsf_points(test_df.dropna(subset=spec.linear), model, scaler, fitted_spec, meta)

        boyce, _ = boyce_quantile_bins(
            test_pred,
            rsf,
            domain,
            n_background_points=n_background_boyce,
            n_bins=n_bins,
            seed=seed + 200 * fold,
        )
        rows.append({"fold": fold, "boyce": boyce, "n_train": len(split.train_thin), "n_test": len(split.test)})

    return pd.DataFrame(rows)


def _select_heldout_individuals(
    ids: Iterable,
    *,
    heldout: str | int | float | Iterable = "all",
    seed: int = 42,
) -> list:
    """Select individuals for leave-one-individual-out validation."""

    ids = pd.Index(pd.Series(list(ids)).dropna().unique()).tolist()

    if heldout == "all":
        return ids

    if isinstance(heldout, int):
        if heldout <= 0:
            raise ValueError("heldout as int must be positive.")
        n = min(heldout, len(ids))

    elif isinstance(heldout, float):
        if not 0 < heldout <= 1:
            raise ValueError("heldout as float must be in (0, 1].")
        n = max(1, int(np.ceil(len(ids) * heldout)))

    else:
        selected = list(heldout)
        missing = sorted(set(selected).difference(ids))
        if missing:
            raise ValueError(f"Held-out IDs not found in data: {missing}")
        return selected

    rng = np.random.default_rng(seed)
    return rng.choice(ids, size=n, replace=False).tolist()


def _domains_by_id(
    used: gpd.GeoDataFrame,
    *,
    id_col: str = "Individual_ID",
    domain: gpd.GeoDataFrame | None = None,
    quantile: float = 0.95,
) -> dict[Any, gpd.GeoDataFrame]:
    """Create one availability domain per individual."""

    domains = {}

    for individual_id, g in used.groupby(id_col):
        if domain is None:
            domains[individual_id] = get_availability_domain(
                g,
                estimator="mcp",
                quantile=quantile,
            )
        else:
            dom = domain.copy()
            if dom.crs != g.crs:
                dom = dom.to_crs(g.crs)
            domains[individual_id] = dom

    return domains


def _extract_predictor_cols(spec: FeatureSpec) -> list[str]:
    cols = []
    cols.extend(getattr(spec, "linear", []) or [])
    cols.extend(getattr(spec, "quadratic", []) or [])
    cols.extend(getattr(spec, "categorical", []) or [])

    for a, b in getattr(spec, "interactions", []) or []:
        cols.extend([a, b])

    return list(dict.fromkeys(cols))

def leave_one_individual_out_rsf(
    reloc: gpd.GeoDataFrame,
    env,
    spec: FeatureSpec,
    *,
    id_col: str = "Individual_ID",
    heldout: str | int | float | Iterable = "all",
    domain: gpd.GeoDataFrame | None = None,
    domain_quantile: float = 0.95,
    thin_train_dt: str | None = "12h",
    thin_test_dt: str | None = None,
    sampling_factor_train: int = 10,
    n_background_boyce: int = 100_000,
    n_bins: int | None = None,
    boyce_window_fraction: float = 0.1,
    boyce_step_fraction: float = 0.02,
    calibration_n_bins: int = 10,
    calibration_n_background: int | None = None,
    calibration_alpha: float = 0.05,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """Leave-one-individual-out RSF validation.

    For each held-out individual, the model is trained on all other individuals,
    with available points sampled within each training individual's own domain.

    Evaluation uses the held-out individual's relocations and availability
    domain. Boyce evaluates ranking relative to random use. Calibration compares
    observed use with fitted static-RSF probability mass across equal-area
    RSF-score quantiles.

    Notes
    -----
    The calibration diagnostic assumes uniform baseline accessibility within
    the held-out individual's availability domain.

    Returns
    -------
    summary:
        Fold-level validation results.
    params:
        Coefficient estimates for every held-out fold.
    boyce_bins:
        Long-format Boyce diagnostics.
    calibration_bins:
        Long-format equal-area RSF-quantile diagnostics.
    diagnostics:
        Fold-specific surfaces, points and validation tables.
    """

    if id_col not in reloc.columns:
        raise KeyError(f"{id_col!r} not found in reloc.")

    if "Timestamp" not in reloc.columns:
        raise KeyError("'Timestamp' column is required.")

    if reloc.crs is None:
        raise ValueError("reloc.crs is None; set a CRS before validation.")

    if calibration_n_bins < 2:
        raise ValueError("calibration_n_bins must be at least 2.")

    if calibration_n_background is None:
        calibration_n_background = n_background_boyce

    # ---------------------------------------------------------
    # Restrict environmental stack to model predictors
    # ---------------------------------------------------------

    predictor_bands = _extract_predictor_cols(spec)

    available_bands = {
        str(band)
        for band in env["band"].values
    }

    missing_bands = [
        band
        for band in predictor_bands
        if band not in available_bands
    ]

    if missing_bands:
        raise ValueError(
            "Predictors required by FeatureSpec are missing "
            f"from env: {missing_bands}"
        )

    env_model = env.sel(
        band=predictor_bands
    )

    diagnostics = {}

    g = reloc.copy()
    g["Timestamp"] = require_timezone_aware(
        g["Timestamp"],
        name="Timestamp",
    )
    g = g.dropna(
        subset=[id_col, "Timestamp", "geometry"],
    ).copy()

    ids_to_holdout = _select_heldout_individuals(
        g[id_col].unique(),
        heldout=heldout,
        seed=seed,
    )

    domains = _domains_by_id(
        g,
        id_col=id_col,
        domain=domain,
        quantile=domain_quantile,
    )

    rows = []
    param_rows = []
    boyce_rows = []
    calibration_rows = []

    for fold_id, heldout_id in enumerate(ids_to_holdout):
        train_used = g.loc[g[id_col] != heldout_id].copy()
        test_used = g.loc[g[id_col] == heldout_id].copy()

        if train_used.empty or test_used.empty:
            rows.append(
                {
                    "fold": fold_id,
                    "heldout_ID": heldout_id,
                    "boyce": np.nan,
                    "n_train_used": len(train_used),
                    "n_test_used": len(test_used),
                    "error": "empty train or test set",
                }
            )
            continue

        # ---------------------------------------------------------
        # Generate training samples separately for each individual
        # ---------------------------------------------------------
        train_parts = []
        n_train_used_fitted = 0

        for j, (train_id, train_i) in enumerate(
            train_used.groupby(id_col)
        ):
            train_i = train_i.copy()

            if thin_train_dt is not None:
                train_i = thin_by_time(
                    train_i,
                    min_dt=thin_train_dt,
                )

            n_train_used_fitted += len(train_i)

            n_available = (
                len(train_i) * sampling_factor_train
            )

            if n_available == 0:
                continue

            samples_i = sample_available_points(
                domains[train_id],
                n_available,
                used=train_i,
                seed=seed + 10_000 * fold_id + j,
                timestamp_col="Timestamp",
            )

            samples_i[id_col] = train_id
            train_parts.append(samples_i)

        if not train_parts:
            rows.append(
                {
                    "fold": fold_id,
                    "heldout_ID": heldout_id,
                    "boyce": np.nan,
                    "n_train_used": len(train_used),
                    "n_test_used": len(test_used),
                    "error": "no training samples generated",
                }
            )
            continue

        train_samples = gpd.GeoDataFrame(
            pd.concat(train_parts, ignore_index=True),
            geometry="geometry",
            crs=g.crs,
        )

        train_df = sample_raster_stack(
            train_samples,
            env_model,
        )

        model, scaler, fitted_spec, meta = fit_rsf(
            train_df,
            spec,
        )

        # ---------------------------------------------------------
        # Store coefficients
        # ---------------------------------------------------------
        coef = model.params.copy()

        bse = getattr(
            model,
            "bse",
            pd.Series(index=coef.index, data=np.nan),
        )

        pvals = getattr(
            model,
            "pvalues",
            pd.Series(index=coef.index, data=np.nan),
        )

        param_row = {
            "fold": fold_id,
            "heldout_ID": heldout_id,
        }

        for name, value in coef.items():
            param_row[f"beta_{name}"] = float(value)

        for name, value in bse.items():
            param_row[f"se_{name}"] = (
                float(value)
                if pd.notna(value)
                else np.nan
            )

        for name, value in pvals.items():
            param_row[f"p_{name}"] = (
                float(value)
                if pd.notna(value)
                else np.nan
            )

        param_rows.append(param_row)

        # ---------------------------------------------------------
        # Predict fold-specific RSF surface
        # ---------------------------------------------------------
        rsf = predict_rsf_surface(
            env_model,
            model,
            scaler,
            fitted_spec,
            meta,
        )

        # ---------------------------------------------------------
        # Evaluate held-out individual
        # ---------------------------------------------------------
        test_eval = test_used.copy()

        if thin_test_dt is not None:
            test_eval = thin_by_time(
                test_eval,
                min_dt=thin_test_dt,
            )

        test_eval["used"] = True

        test_df = sample_raster_stack(
            test_eval,
            env_model,
        )

        test_df = test_df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        predictor_cols = _extract_predictor_cols(
            fitted_spec
        )
        predictor_cols = [
            col
            for col in predictor_cols
            if col in test_df.columns
        ]

        test_df = test_df.dropna(
            subset=predictor_cols,
        )

        if test_df.empty:
            rows.append(
                {
                    "fold": fold_id,
                    "heldout_ID": heldout_id,
                    "boyce": np.nan,
                    "n_train_used": int(len(train_used)),
                    "n_train_used_fitted": int(
                        n_train_used_fitted
                    ),
                    "n_train_samples": int(
                        len(train_samples)
                    ),
                    "n_test_used": int(len(test_used)),
                    "n_test_eval": 0,
                    "error": (
                        "no test points with complete "
                        "environmental predictors"
                    ),
                }
            )
            continue

        test_pred = predict_rsf_points(
            test_df,
            model,
            scaler,
            fitted_spec,
            meta,
        )

        # Safeguard in case predict_rsf_points does not preserve metadata.
        if "used" not in test_pred.columns:
            test_pred["used"] = True

        # ---------------------------------------------------------
        # Boyce validation
        # ---------------------------------------------------------
        if n_bins is None:
            boyce, bins = boyce_sliding_window(
                test_pred,
                rsf,
                domains[heldout_id],
                n_background_points=n_background_boyce,
                window_fraction=boyce_window_fraction,
                step_fraction=boyce_step_fraction,
                seed=seed + 20_000 * fold_id,
            )
        else:
            boyce, bins = boyce_quantile_bins(
                test_pred,
                rsf,
                domains[heldout_id],
                n_background_points=n_background_boyce,
                n_bins=n_bins,
                seed=seed + 20_000 * fold_id,
            )

        bins = bins.copy()
        bins["heldout_ID"] = heldout_id
        bins["fold"] = fold_id
        bins["boyce"] = float(boyce)
        boyce_rows.append(bins)

        # ---------------------------------------------------------
        # Equal-area RSF-quantile calibration
        # ---------------------------------------------------------
        calibration = calibration_rsf_quantile_bins(
            pred=test_pred,
            rsf=rsf,
            domain=domains[heldout_id],
            n_background_points=calibration_n_background,
            n_bins=calibration_n_bins,
            seed=seed + 30_000 * fold_id,
            pred_col="rsf_pred",
            alpha=calibration_alpha,
        )

        calibration = calibration.copy()
        calibration["heldout_ID"] = heldout_id
        calibration["fold"] = fold_id
        calibration_rows.append(calibration)

        rows.append(
            {
                "fold": fold_id,
                "heldout_ID": heldout_id,
                "boyce": float(boyce),
                "n_train_used": int(len(train_used)),
                "n_train_used_fitted": int(
                    n_train_used_fitted
                ),
                "n_train_samples": int(
                    len(train_samples)
                ),
                "n_test_used": int(len(test_used)),
                "n_test_after_thinning": int(
                    len(test_eval)
                ),
                "n_test_eval": int(len(test_df)),
                "error": None,
            }
        )

        diagnostics[heldout_id] = {
            "model": model,
            "scaler": scaler,
            "spec": fitted_spec,
            "meta": meta,
            "rsf": rsf,
            "domain": domains[heldout_id],
            "test_points": test_eval.copy(),
            "test_pred": test_pred.copy(),
            "boyce_bins": bins.copy(),
            "calibration_bins": calibration.copy(),
        }

    summary = pd.DataFrame(rows)
    params = pd.DataFrame(param_rows)

    boyce_bins = (
        pd.concat(
            boyce_rows,
            ignore_index=True,
        )
        if boyce_rows
        else pd.DataFrame()
    )

    calibration_bins = (
        pd.concat(
            calibration_rows,
            ignore_index=True,
        )
        if calibration_rows
        else pd.DataFrame()
    )

    return (
        summary,
        params,
        boyce_bins,
        calibration_bins,
        diagnostics,
    )