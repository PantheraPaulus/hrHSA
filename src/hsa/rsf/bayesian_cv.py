"""Cross-validation workflows for hierarchical Bayesian RSFs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from hsa._time import require_timezone_aware

from hsa.rsf.bayesian import build_bayesian_rsf_model, prepare_bayesian_rsf_data
from hsa.rsf.bayesian_validation import (
    bayesian_boyce_quantile_scores,
    prepare_bayesian_boyce_scores,
)
from hsa.rsf.cv import _domains_by_id, _select_heldout_individuals, thin_by_time
from hsa.sampling import sample_available_points, sample_raster_stack


def _sample_frame(*args, **kwargs) -> pd.DataFrame:
    """Return the dataframe across old/new sample_raster_stack return contracts."""
    sampled = sample_raster_stack(*args, **kwargs)
    return sampled[0] if isinstance(sampled, tuple) else sampled


def _select_predictor_env(env, predictors: Sequence[str]):
    """Return a lazy raster view containing only fitted predictor bands."""
    predictors = list(dict.fromkeys(predictors))
    if "band" not in env.dims:
        raise ValueError("env must contain a 'band' dimension.")

    available = {str(value) for value in env["band"].values}
    missing = [
        predictor
        for predictor in predictors
        if predictor not in available
    ]
    if missing:
        raise ValueError(
            "Predictors required by the Bayesian RSF are missing from env: "
            f"{missing}"
        )

    return env.sel(band=predictors)


def _posterior_beta_summary(
    idata,
    *,
    predictors: Sequence[str],
    ci_prob: float,
    fold: int,
    heldout_id,
) -> list[dict[str, Any]]:
    """Return a tidy population-beta summary for one outer LOIO fold."""
    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ArviZ is required; install hsa[bayesian].") from exc

    beta = idata.posterior["beta"]
    alpha = (1.0 - ci_prob) / 2.0
    rows: list[dict[str, Any]] = []

    for predictor in predictors:
        values = np.asarray(
            beta.sel(predictor=predictor).values,
            dtype=float,
        ).ravel()
        hdi = np.asarray(
            az.hdi(values, prob=ci_prob),
            dtype=float,
        )
        rows.append(
            {
                "fold": fold,
                "heldout_ID": heldout_id,
                "predictor": predictor,
                "beta_mean": float(np.mean(values)),
                "beta_sd": float(np.std(values, ddof=1)),
                "beta_median": float(np.median(values)),
                "beta_lower": float(hdi[0]),
                "beta_upper": float(hdi[1]),
                "p_beta_gt_zero": float(np.mean(values > 0)),
                "odds_ratio_median": float(np.exp(np.median(values))),
                "eti_lower": float(np.quantile(values, alpha)),
                "eti_upper": float(np.quantile(values, 1.0 - alpha)),
            }
        )
    return rows


def _sampling_diagnostics(idata) -> dict[str, Any]:
    """Lightweight sampler diagnostics for a fold without running PPC."""
    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ArviZ is required; install hsa[bayesian].") from exc

    posterior = idata.posterior
    variables = [
        variable
        for variable in (
            "alpha",
            "beta",
            "sigma_alpha",
            "sigma_beta",
            "hs_tau",
            "hs_lambda",
            "hs_c2",
            "hs_beta_raw",
        )
        if variable in posterior
    ]
    diag = az.summary(
        idata,
        var_names=variables,
        kind="diagnostics",
    )

    n_divergences = 0
    if "diverging" in idata.sample_stats:
        n_divergences = int(idata.sample_stats["diverging"].sum().item())

    return {
        "max_rhat": (
            float(diag["r_hat"].max())
            if "r_hat" in diag
            else np.nan
        ),
        "min_ess_bulk": (
            float(diag["ess_bulk"].min())
            if "ess_bulk" in diag
            else np.nan
        ),
        "min_ess_tail": (
            float(diag["ess_tail"].min())
            if "ess_tail" in diag
            else np.nan
        ),
        "n_divergences": n_divergences,
    }


def leave_one_individual_out_bayesian_rsf(
    reloc: gpd.GeoDataFrame,
    env,
    *,
    predictors: Sequence[str],
    binning: Mapping[str, float | None] | None = None,
    id_col: str = "individual-local-identifier",
    heldout: str | int | float | Iterable = "all",
    domain: gpd.GeoDataFrame | None = None,
    domain_quantile: float = 0.95,
    thin_train_dt: str | None = None,
    thin_test_dt: str | None = None,
    sampling_factor_train: int = 50,
    random_intercept: bool = True,
    random_slopes: Sequence[str] | None = None,
    n_background_boyce: int = 100_000,
    n_boyce_draws: int = 500,
    n_bins: int = 20,
    ci_prob: float = 0.95,
    model_kwargs: Mapping[str, Any] | None = None,
    sample_kwargs: Mapping[str, Any] | None = None,
    store_scores: bool = True,
    fail_fast: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Leave-one-individual-out validation for a hierarchical Bayesian RSF.

    Each outer fold is fitted using all individuals except the held-out animal.
    Training availability is sampled separately inside every training animal's
    own availability domain. Only the fitted predictor bands are sampled from
    the environmental stack. The held-out animal is evaluated against its own
    domain using a fixed set of background coordinates across posterior draws.

    The returned ``diagnostics`` stores the Boyce score matrices by default so
    post-fit blocked-bootstrap and contiguous temporal validation can be rerun
    without refitting the expensive Bayesian model.
    """
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyMC is required; install hsa[bayesian].") from exc

    predictors = list(dict.fromkeys(predictors))
    random_slopes = [] if random_slopes is None else list(random_slopes)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    sample_kwargs = {} if sample_kwargs is None else dict(sample_kwargs)

    if id_col not in reloc.columns:
        raise KeyError(f"{id_col!r} not found in reloc.")
    if "Timestamp" not in reloc.columns:
        raise KeyError("'Timestamp' column is required.")
    if reloc.crs is None:
        raise ValueError("reloc.crs is None; set a CRS before validation.")
    if not predictors:
        raise ValueError("At least one predictor is required.")
    if sampling_factor_train <= 0:
        raise ValueError("sampling_factor_train must be positive.")
    if n_background_boyce <= 0:
        raise ValueError("n_background_boyce must be positive.")

    env_model = _select_predictor_env(env, predictors)

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

    summary_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    boyce_rows: list[pd.DataFrame] = []
    diagnostics: dict[Any, Any] = {}

    default_sample_kwargs: dict[str, Any] = {
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.9,
        "return_inferencedata": True,
    }
    default_sample_kwargs.update(sample_kwargs)

    for fold, heldout_id in enumerate(ids_to_holdout):
        print(
            f"[Bayesian LOIO {fold + 1}/{len(ids_to_holdout)}] "
            f"Held out: {heldout_id}"
        )
        try:
            train_used = g.loc[g[id_col] != heldout_id].copy()
            test_used = g.loc[g[id_col] == heldout_id].copy()
            if train_used.empty or test_used.empty:
                raise ValueError("empty train or held-out set")

            train_parts: list[gpd.GeoDataFrame] = []
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
                n_available = len(train_i) * sampling_factor_train
                if n_available == 0:
                    continue

                sampled_i = sample_available_points(
                    domains[train_id],
                    n_available,
                    used=train_i,
                    seed=seed + 10_000 * fold + j,
                    timestamp_col="Timestamp",
                )
                sampled_i[id_col] = train_id
                train_parts.append(sampled_i)

            if not train_parts:
                raise ValueError("no training samples generated")

            train_samples = gpd.GeoDataFrame(
                pd.concat(train_parts, ignore_index=True),
                geometry="geometry",
                crs=g.crs,
            )

            train_df = (
                _sample_frame(
                    train_samples,
                    env_model,
                    bands=predictors,
                    id_cols=id_col,
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=predictors)
            )

            required_columns = [
                id_col,
                "used",
                *predictors,
            ]
            missing = [
                column
                for column in required_columns
                if column not in train_df.columns
            ]
            if missing:
                raise RuntimeError(
                    "Columns were lost while constructing Bayesian "
                    f"training data: {missing}"
                )

            bayes_data = prepare_bayesian_rsf_data(
                train_df,
                predictors=predictors,
                binning=None if binning is None else dict(binning),
                id_col=id_col,
            )
            model = build_bayesian_rsf_model(
                bayes_data,
                predictors=predictors,
                random_intercept=random_intercept,
                random_slopes=random_slopes,
                **model_kwargs,
            )

            fold_seed = seed + 100_000 * fold
            with model:
                idata = pm.sample(
                    random_seed=fold_seed,
                    **default_sample_kwargs,
                )

            param_rows.extend(
                _posterior_beta_summary(
                    idata,
                    predictors=predictors,
                    ci_prob=ci_prob,
                    fold=fold,
                    heldout_id=heldout_id,
                )
            )
            sampler_diag = _sampling_diagnostics(idata)

            test_eval = test_used.copy()
            if thin_test_dt is not None:
                test_eval = thin_by_time(
                    test_eval,
                    min_dt=thin_test_dt,
                )

            scores = prepare_bayesian_boyce_scores(
                used=test_eval,
                env=env_model,
                idata=idata,
                meta=bayes_data["meta"],
                predictors=predictors,
                domain=domains[heldout_id],
                n_background=n_background_boyce,
                n_draws=n_boyce_draws,
                seed=fold_seed + 50_000,
                exponentiate=False,
            )
            boyce = bayesian_boyce_quantile_scores(
                scores["used_scores"],
                scores["available_scores"],
                n_bins=n_bins,
                ci_prob=ci_prob,
            )

            bdraws = np.asarray(
                boyce["boyce_draws"],
                dtype=float,
            )
            finite_bdraws = bdraws[np.isfinite(bdraws)]
            boyce_summary = boyce["boyce_summary"]

            summary_rows.append(
                {
                    "fold": fold,
                    "heldout_ID": heldout_id,
                    "boyce_mean": (
                        float(np.mean(finite_bdraws))
                        if len(finite_bdraws)
                        else np.nan
                    ),
                    "boyce_median": boyce_summary["median"],
                    "boyce_lower": boyce_summary["lower"],
                    "boyce_upper": boyce_summary["upper"],
                    "p_boyce_gt_zero": boyce_summary["p_gt_zero"],
                    "n_train_used": int(len(train_used)),
                    "n_train_used_fitted": int(n_train_used_fitted),
                    "n_train_samples": int(len(train_samples)),
                    "n_test_used": int(len(test_used)),
                    "n_test_eval": int(len(scores["used_data"])),
                    "n_background": int(len(scores["available_data"])),
                    "n_posterior_draws": int(len(finite_bdraws)),
                    **sampler_diag,
                    "error": None,
                }
            )

            curve = boyce["curve_summary"].copy()
            curve["fold"] = fold
            curve["heldout_ID"] = heldout_id
            curve["boyce_median"] = boyce_summary["median"]
            curve["boyce_lower"] = boyce_summary["lower"]
            curve["boyce_upper"] = boyce_summary["upper"]
            boyce_rows.append(curve)

            diagnostics[heldout_id] = {
                "fold": fold,
                "model": model,
                "idata": idata,
                "bayes_data": bayes_data,
                "domain": domains[heldout_id],
                "train_used": train_used,
                "test_used": test_used,
                "test_eval": test_eval,
                "boyce": boyce,
                "env": env_model,
                "sampler": sampler_diag,
            }
            if store_scores:
                diagnostics[heldout_id]["scores"] = scores

        except Exception as exc:
            if fail_fast:
                raise

            summary_rows.append(
                {
                    "fold": fold,
                    "heldout_ID": heldout_id,
                    "boyce_mean": np.nan,
                    "boyce_median": np.nan,
                    "boyce_lower": np.nan,
                    "boyce_upper": np.nan,
                    "p_boyce_gt_zero": np.nan,
                    "n_train_used": np.nan,
                    "n_train_used_fitted": np.nan,
                    "n_train_samples": np.nan,
                    "n_test_used": np.nan,
                    "n_test_eval": np.nan,
                    "n_background": np.nan,
                    "n_posterior_draws": np.nan,
                    "max_rhat": np.nan,
                    "min_ess_bulk": np.nan,
                    "min_ess_tail": np.nan,
                    "n_divergences": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            diagnostics[heldout_id] = {
                "fold": fold,
                "error": exc,
            }

    summary = pd.DataFrame(summary_rows)
    params = pd.DataFrame(param_rows)
    boyce_bins = (
        pd.concat(boyce_rows, ignore_index=True)
        if boyce_rows
        else pd.DataFrame()
    )
    return summary, params, boyce_bins, diagnostics


__all__ = ["leave_one_individual_out_bayesian_rsf"]
