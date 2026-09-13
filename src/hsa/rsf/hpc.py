"""HPC-oriented RSF validation kernels.

These functions deliberately live beside the original serial CV implementation.
The serial path remains the scientific reference; the prepared path reuses
immutable sampled predictors and can distribute independent held-out folds.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hsa.compute.prepared import PreparedDataset
from hsa.rsf.cv import _extract_predictor_cols, _select_heldout_individuals
from hsa.rsf.cv_parallel import execute_fold_calls
from hsa.rsf.model import fit_rsf, predict_rsf_points
from hsa.rsf.surface import predict_rsf_surface
from hsa.rsf.surface_fast import predict_rsf_surface_chunked
from hsa.rsf.validation import boyce_quantile_bins, calibration_rsf_quantile_bins


def _inner_dask_context():
    """Use a local synchronous scheduler when already inside a Dask worker.

    LOIO folds are the distributed unit. A fold may still touch a storage-backed
    Dask raster during surface validation. Submitting that inner graph back to the
    same distributed scheduler can create nested scheduling/deadlock and excessive
    communication. Inside a worker we therefore execute the small inner graph
    synchronously in that worker process. The graph still reads only the chunks it
    needs from shared storage.
    """
    try:
        from dask.distributed import get_worker

        get_worker()
    except Exception:
        return nullcontext()

    try:
        import dask

        return dask.config.set(scheduler="synchronous")
    except Exception:
        return nullcontext()


def _fit_fold_model(
    train_df,
    spec,
    *,
    fit_method: str,
    fit_kwargs: Mapping[str, Any] | None,
    blas_threads: int,
):
    """Fit one frequentist fold under an explicit native-thread budget."""
    if blas_threads <= 0:
        raise ValueError("blas_threads must be positive.")
    with threadpool_limits(limits=blas_threads):
        return fit_rsf(
            train_df,
            spec,
            method=fit_method,
            fit_kwargs=fit_kwargs,
        )


def _prepared_loio_fold(
    *,
    prepared_root: str,
    heldout_id: Any,
    fold_id: int,
    env_model,
    domain,
    spec,
    n_background: int,
    n_bins: int,
    calibration_n_bins: int,
    calibration_n_background: int,
    calibration_alpha: float,
    seed: int,
    surface_engine: str,
    target_chunk_mb: int,
    keep_models: bool,
    keep_surfaces: bool,
    fit_method: str,
    fit_kwargs: Mapping[str, Any] | None,
    blas_threads: int,
) -> dict[str, Any]:
    """Execute one prepared LOIO fold; designed to be Dask-picklable."""
    prepared = PreparedDataset.open(prepared_root)
    train_df = prepared.load(exclude=heldout_id)
    test_all = prepared.load(include=[heldout_id])

    if train_df.empty or test_all.empty:
        return {
            "row": {
                "fold": fold_id,
                "heldout_ID": heldout_id,
                "boyce": np.nan,
                "n_train_samples": int(len(train_df)),
                "n_test_eval": 0,
                "optimizer": fit_method,
                "blas_threads": blas_threads,
                "error": "empty prepared train or test partition",
            },
            "param": None,
            "boyce_bins": None,
            "calibration_bins": None,
            "diagnostics": {},
        }

    model, scaler, fitted_spec, meta = _fit_fold_model(
        train_df,
        spec,
        fit_method=fit_method,
        fit_kwargs=fit_kwargs,
        blas_threads=blas_threads,
    )
    test_df = test_all.loc[test_all["used"].astype(bool)].copy()
    predictor_cols = [
        value
        for value in _extract_predictor_cols(fitted_spec)
        if value in test_df.columns
    ]
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=predictor_cols
    )
    if test_df.empty:
        return {
            "row": {
                "fold": fold_id,
                "heldout_ID": heldout_id,
                "boyce": np.nan,
                "n_train_samples": int(len(train_df)),
                "n_test_eval": 0,
                "optimizer": fit_method,
                "blas_threads": blas_threads,
                "error": "no complete held-out predictor rows",
            },
            "param": None,
            "boyce_bins": None,
            "calibration_bins": None,
            "diagnostics": {},
        }

    test_pred = predict_rsf_points(
        test_df,
        model,
        scaler,
        fitted_spec,
        meta,
    )
    if "used" not in test_pred:
        test_pred["used"] = True

    with _inner_dask_context():
        if surface_engine == "chunked":
            rsf = predict_rsf_surface_chunked(
                env_model,
                model,
                scaler,
                fitted_spec,
                meta,
                target_chunk_mb=target_chunk_mb,
            )
        elif surface_engine == "reference":
            rsf = predict_rsf_surface(
                env_model,
                model,
                scaler,
                fitted_spec,
                meta,
            )
        else:
            raise ValueError("surface_engine must be 'reference' or 'chunked'.")

        boyce, bins = boyce_quantile_bins(
            test_pred,
            rsf,
            domain,
            n_background_points=n_background,
            n_bins=n_bins,
            seed=seed + 20_000 * fold_id,
        )
        bins = bins.copy()
        bins["heldout_ID"] = heldout_id
        bins["fold"] = fold_id
        bins["boyce"] = float(boyce) if np.isfinite(boyce) else np.nan

        calibration = calibration_rsf_quantile_bins(
            pred=test_pred,
            rsf=rsf,
            domain=domain,
            n_background_points=calibration_n_background,
            n_bins=calibration_n_bins,
            seed=seed + 30_000 * fold_id,
            pred_col="rsf_pred",
            alpha=calibration_alpha,
        ).copy()
        calibration["heldout_ID"] = heldout_id
        calibration["fold"] = fold_id

    coef = pd.Series(model.params)
    bse = pd.Series(getattr(model, "bse", np.nan), index=coef.index)
    pvalues = pd.Series(getattr(model, "pvalues", np.nan), index=coef.index)
    param = {"fold": fold_id, "heldout_ID": heldout_id}
    for name, value in coef.items():
        param[f"beta_{name}"] = float(value)
    for name, value in bse.items():
        param[f"se_{name}"] = float(value) if pd.notna(value) else np.nan
    for name, value in pvalues.items():
        param[f"p_{name}"] = float(value) if pd.notna(value) else np.nan

    diagnostics: dict[str, Any] = {
        "scaler": scaler,
        "spec": fitted_spec,
        "meta": meta,
        "domain": domain,
        "test_pred": test_pred,
        "boyce_bins": bins,
        "calibration_bins": calibration,
        "execution": {
            "optimizer": fit_method,
            "blas_threads": blas_threads,
        },
    }
    if keep_models:
        diagnostics["model"] = model
    if keep_surfaces:
        diagnostics["rsf"] = rsf

    return {
        "row": {
            "fold": fold_id,
            "heldout_ID": heldout_id,
            "boyce": float(boyce) if np.isfinite(boyce) else np.nan,
            "n_train_samples": int(len(train_df)),
            "n_train_used_fitted": int(train_df["used"].astype(bool).sum()),
            "n_test_used": int(test_all["used"].astype(bool).sum()),
            "n_test_eval": int(len(test_df)),
            "prepared": True,
            "optimizer": fit_method,
            "blas_threads": blas_threads,
            "error": None,
        },
        "param": param,
        "boyce_bins": bins,
        "calibration_bins": calibration,
        "diagnostics": diagnostics,
    }


def leave_one_individual_out_rsf_prepared(
    analysis,
    prepared: PreparedDataset | str | Path,
    spec,
    *,
    heldout: str | int | float | list[Any] = "all",
    n_background: int = 100_000,
    n_bins: int = 20,
    calibration_n_bins: int = 10,
    calibration_n_background: int | None = None,
    calibration_alpha: float = 0.05,
    seed: int = 42,
    client=None,
    surface_engine: str = "chunked",
    target_chunk_mb: int = 256,
    keep_models: bool = False,
    keep_surfaces: bool = False,
    fit_method: str = "newton",
    fit_kwargs: Mapping[str, Any] | None = None,
    blas_threads: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run LOIO from cached individual partitions, optionally across Dask workers.

    The expensive environmental extraction performed by the reference LOIO path
    is absent here. Each fold reads all training partitions except its held-out
    individual from shared storage, fits independently, and validates against the
    held-out domain. Passing a Dask client turns the folds into embarrassingly
    parallel tasks without changing the scientific fold definition.

    Fold tasks are the outer distributed unit. The default ``blas_threads=1`` is
    intentional: workstation benchmarks showed little benefit from spending many
    native threads on one low-dimensional logistic fit. Use ``fit_method='lbfgs'``
    as the high-throughput large-data option after validating it against Newton.
    """
    if blas_threads <= 0:
        raise ValueError("blas_threads must be positive.")
    if not isinstance(prepared, PreparedDataset):
        prepared = PreparedDataset.open(prepared)
    if prepared.id_col != analysis.id_col:
        raise ValueError(
            f"Prepared id_col={prepared.id_col!r} does not match "
            f"analysis id_col={analysis.id_col!r}."
        )

    requested_predictors = set(_extract_predictor_cols(spec))
    missing_prepared = sorted(requested_predictors.difference(prepared.predictors))
    if missing_prepared:
        raise ValueError(
            "Prepared dataset is missing predictors required by FeatureSpec: "
            f"{missing_prepared}"
        )

    predictor_bands = list(dict.fromkeys(_extract_predictor_cols(spec)))
    env_model = analysis.env.sel(band=predictor_bands)
    ids = _select_heldout_individuals(
        prepared.individuals,
        heldout=heldout,
        seed=seed,
    )
    calibration_n_background = (
        n_background
        if calibration_n_background is None
        else calibration_n_background
    )

    calls = [
        dict(
            prepared_root=str(prepared.root),
            heldout_id=individual_id,
            fold_id=fold_id,
            env_model=env_model,
            domain=analysis.domains[individual_id],
            spec=spec,
            n_background=n_background,
            n_bins=n_bins,
            calibration_n_bins=calibration_n_bins,
            calibration_n_background=calibration_n_background,
            calibration_alpha=calibration_alpha,
            seed=seed,
            surface_engine=surface_engine,
            target_chunk_mb=target_chunk_mb,
            keep_models=keep_models,
            keep_surfaces=keep_surfaces,
            fit_method=fit_method,
            fit_kwargs=None if fit_kwargs is None else dict(fit_kwargs),
            blas_threads=blas_threads,
        )
        for fold_id, individual_id in enumerate(ids)
    ]

    results = execute_fold_calls(_prepared_loio_fold, calls, client=client)

    rows = [result["row"] for result in results]
    params = [
        result["param"] for result in results if result["param"] is not None
    ]
    boyce = [
        result["boyce_bins"]
        for result in results
        if result["boyce_bins"] is not None and not result["boyce_bins"].empty
    ]
    calibration = [
        result["calibration_bins"]
        for result in results
        if result["calibration_bins"] is not None
        and not result["calibration_bins"].empty
    ]
    diagnostics = {
        result["row"]["heldout_ID"]: result["diagnostics"]
        for result in results
        if result["diagnostics"]
    }
    return (
        pd.DataFrame(rows),
        pd.DataFrame(params),
        pd.concat(boyce, ignore_index=True) if boyce else pd.DataFrame(),
        pd.concat(calibration, ignore_index=True)
        if calibration
        else pd.DataFrame(),
        diagnostics,
    )
