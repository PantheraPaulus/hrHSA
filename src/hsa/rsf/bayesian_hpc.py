"""Prepared and fold-parallel cross-validation for hierarchical Bayesian RSFs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hsa.compute.prepared import PreparedDataset
from hsa.rsf.bayesian import build_bayesian_rsf_model, prepare_bayesian_rsf_data
from hsa.rsf.bayesian_cv import (
    _posterior_beta_summary,
    _sampling_diagnostics,
    _select_predictor_env,
)
from hsa.rsf.bayesian_validation import (
    bayesian_boyce_quantile_scores,
    prepare_bayesian_boyce_scores,
)
from hsa.rsf.cv import _select_heldout_individuals, thin_by_time
from hsa.rsf.cv_parallel import execute_fold_calls
from hsa.rsf.hpc import _inner_dask_context


def _numpy_dataset(dataset):
    """Return a Dataset with all data variables materialized as host NumPy arrays."""
    if dataset is None:
        return None
    out = dataset.copy(deep=False)
    for name, value in dataset.data_vars.items():
        out[name] = value.copy(data=np.asarray(value.data))
    return out


def _materialize_inference_tree(idata):
    """Synchronize lazy/JAX inference output and return a compact host DataTree.

    BlackJAX/JAX may return before queued CPU work has completed. Explicitly
    converting posterior and sample statistics to NumPy makes fold completion a
    real synchronization point and avoids shipping device-backed arrays between
    Dask workers and the driver.
    """
    groups = {"posterior": _numpy_dataset(idata.posterior)}
    sample_stats = _numpy_dataset(getattr(idata, "sample_stats", None))
    if sample_stats is not None:
        groups["sample_stats"] = sample_stats
    return xr.DataTree.from_dict(groups)


def _sampling_options(
    sample_kwargs: Mapping[str, Any] | None,
    *,
    distributed: bool,
) -> dict[str, Any]:
    """Build fold sampling options while avoiding nested process oversubscription."""
    options: dict[str, Any] = {
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.9,
        "return_inferencedata": True,
    }
    if sample_kwargs is not None:
        options.update(dict(sample_kwargs))

    # The Dask fold is the outer parallel unit. Do not let every worker spawn a
    # second four-process pool unless the caller explicitly asks for that layout.
    if distributed:
        options.setdefault("cores", 1)

    # Fold seeds are assigned deterministically below.
    options.pop("random_seed", None)
    return options


def _error_result(fold_id: int, heldout_id, exc: BaseException) -> dict[str, Any]:
    return {
        "row": {
            "fold": fold_id,
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
            "prepared": True,
            "error": f"{type(exc).__name__}: {exc}",
        },
        "params": [],
        "boyce_bins": None,
        "diagnostics": {"fold": fold_id, "error": f"{type(exc).__name__}: {exc}"},
    }


def _prepared_bayesian_loio_fold(
    *,
    prepared_root: str,
    heldout_id: Any,
    fold_id: int,
    test_used,
    env_model,
    domain,
    predictors: Sequence[str],
    binning: Mapping[str, float | None] | None,
    id_col: str,
    thin_test_dt: str | None,
    random_intercept: bool,
    random_slopes: Sequence[str],
    n_background: int,
    n_boyce_draws: int,
    n_bins: int,
    ci_prob: float,
    model_kwargs: Mapping[str, Any],
    sample_kwargs: Mapping[str, Any],
    outer_parallel: bool,
    store_scores: bool,
    keep_idata: bool,
    keep_model: bool,
    keep_bayes_data: bool,
    fail_fast: bool,
    seed: int,
) -> dict[str, Any]:
    """Execute one prepared Bayesian LOIO fold inside one worker process."""
    try:
        import pymc as pm

        prepared = PreparedDataset.open(prepared_root)
        train_df = prepared.load(exclude=heldout_id)
        if train_df.empty or test_used.empty:
            raise ValueError("empty prepared training or held-out set")

        required = [id_col, "used", *predictors]
        missing = [column for column in required if column not in train_df.columns]
        if missing:
            raise RuntimeError(f"Prepared Bayesian training columns missing: {missing}")

        bayes_data = prepare_bayesian_rsf_data(
            train_df,
            predictors=list(predictors),
            binning=None if binning is None else dict(binning),
            id_col=id_col,
        )
        model = build_bayesian_rsf_model(
            bayes_data,
            predictors=list(predictors),
            random_intercept=random_intercept,
            random_slopes=list(random_slopes),
            **dict(model_kwargs),
        )

        sampling = dict(sample_kwargs)
        # With outer fold parallelism, keep native BLAS work within one core by
        # default. Serial prepared CV keeps PyMC's normal behavior unless the
        # caller supplied an explicit blas_cores value.
        if outer_parallel and "blas_cores" in signature(pm.sample).parameters:
            sampling.setdefault("blas_cores", 1)

        fold_seed = seed + 100_000 * fold_id
        with model:
            raw_idata = pm.sample(random_seed=fold_seed, **sampling)

        # This is deliberately inside the fold task. It is both the JAX
        # synchronization point and a serialization boundary for Dask.
        idata = _materialize_inference_tree(raw_idata)
        del raw_idata

        params = _posterior_beta_summary(
            idata,
            predictors=predictors,
            ci_prob=ci_prob,
            fold=fold_id,
            heldout_id=heldout_id,
        )
        sampler_diag = _sampling_diagnostics(idata)

        test_eval = test_used.copy()
        if thin_test_dt is not None:
            test_eval = thin_by_time(test_eval, min_dt=thin_test_dt)

        with _inner_dask_context():
            scores = prepare_bayesian_boyce_scores(
                used=test_eval,
                env=env_model,
                idata=idata,
                meta=bayes_data["meta"],
                predictors=list(predictors),
                domain=domain,
                n_background=n_background,
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

        draws = np.asarray(boyce["boyce_draws"], dtype=float)
        finite_draws = draws[np.isfinite(draws)]
        boyce_summary = boyce["boyce_summary"]
        curve = boyce["curve_summary"].copy()
        curve["fold"] = fold_id
        curve["heldout_ID"] = heldout_id
        curve["boyce_median"] = boyce_summary["median"]
        curve["boyce_lower"] = boyce_summary["lower"]
        curve["boyce_upper"] = boyce_summary["upper"]

        diagnostics: dict[str, Any] = {
            "fold": fold_id,
            "domain": domain,
            "boyce": boyce,
            "sampler": sampler_diag,
        }
        if store_scores:
            diagnostics["scores"] = scores
        if keep_idata:
            diagnostics["idata"] = idata
        if keep_model:
            diagnostics["model"] = model
        if keep_bayes_data:
            diagnostics["bayes_data"] = bayes_data

        n_train_used = int(train_df["used"].astype(bool).sum())
        return {
            "row": {
                "fold": fold_id,
                "heldout_ID": heldout_id,
                "boyce_mean": (
                    float(np.mean(finite_draws)) if len(finite_draws) else np.nan
                ),
                "boyce_median": boyce_summary["median"],
                "boyce_lower": boyce_summary["lower"],
                "boyce_upper": boyce_summary["upper"],
                "p_boyce_gt_zero": boyce_summary["p_gt_zero"],
                "n_train_used": n_train_used,
                "n_train_used_fitted": n_train_used,
                "n_train_samples": int(len(train_df)),
                "n_test_used": int(len(test_used)),
                "n_test_eval": int(len(scores["used_data"])),
                "n_background": int(len(scores["available_data"])),
                "n_posterior_draws": int(len(finite_draws)),
                **sampler_diag,
                "prepared": True,
                "error": None,
            },
            "params": params,
            "boyce_bins": curve,
            "diagnostics": diagnostics,
        }
    except Exception as exc:
        if fail_fast:
            raise
        return _error_result(fold_id, heldout_id, exc)


def leave_one_individual_out_bayesian_rsf_prepared(
    analysis,
    prepared: PreparedDataset | str | Path,
    *,
    predictors: Sequence[str],
    binning: Mapping[str, float | None] | None = None,
    heldout: str | int | float | Sequence[Any] = "all",
    thin_test_dt: str | None = None,
    random_intercept: bool = True,
    random_slopes: Sequence[str] | None = None,
    n_background: int = 100_000,
    n_boyce_draws: int = 500,
    n_bins: int = 20,
    ci_prob: float = 0.95,
    model_kwargs: Mapping[str, Any] | None = None,
    sample_kwargs: Mapping[str, Any] | None = None,
    client=None,
    store_scores: bool = False,
    keep_idata: bool = False,
    keep_model: bool = False,
    keep_bayes_data: bool = False,
    fail_fast: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run prepared Bayesian LOIO with independent folds as the parallel unit.

    Training availability and environmental extraction are loaded from the
    prepared per-individual Parquet cache. Only held-out evaluation/background
    sampling remains fold-specific. Passing ``client`` submits one fold per Dask
    task and defaults PyMC's inner ``cores`` setting to one unless explicitly
    overridden, preventing nested process oversubscription.

    Large fold state is deliberately opt-in. ``store_scores=False`` and the
    ``keep_*`` flags default to compact results suitable for many concurrent folds.
    Enable them when post-fit uncertainty or per-fold posterior inspection is
    required.
    """
    if not isinstance(prepared, PreparedDataset):
        prepared = PreparedDataset.open(prepared)
    if prepared.id_col != analysis.id_col:
        raise ValueError(
            f"Prepared id_col={prepared.id_col!r} does not match "
            f"analysis id_col={analysis.id_col!r}."
        )

    predictors = list(dict.fromkeys(predictors))
    missing = sorted(set(predictors).difference(prepared.predictors))
    if missing:
        raise ValueError(f"Prepared dataset is missing Bayesian predictors: {missing}")

    ids = _select_heldout_individuals(
        prepared.individuals,
        heldout=heldout,
        seed=seed,
    )
    env_model = _select_predictor_env(analysis.env, predictors)
    random_slopes = [] if random_slopes is None else list(random_slopes)
    outer_parallel = client is not None
    sampling = _sampling_options(sample_kwargs, distributed=outer_parallel)

    calls = [
        {
            "prepared_root": str(prepared.root),
            "heldout_id": heldout_id,
            "fold_id": fold_id,
            "test_used": analysis.reloc.loc[
                analysis.reloc[analysis.id_col] == heldout_id
            ].copy(),
            "env_model": env_model,
            "domain": analysis.domains[heldout_id],
            "predictors": predictors,
            "binning": None if binning is None else dict(binning),
            "id_col": analysis.id_col,
            "thin_test_dt": thin_test_dt,
            "random_intercept": random_intercept,
            "random_slopes": random_slopes,
            "n_background": n_background,
            "n_boyce_draws": n_boyce_draws,
            "n_bins": n_bins,
            "ci_prob": ci_prob,
            "model_kwargs": {} if model_kwargs is None else dict(model_kwargs),
            "sample_kwargs": sampling,
            "outer_parallel": outer_parallel,
            "store_scores": store_scores,
            "keep_idata": keep_idata,
            "keep_model": keep_model,
            "keep_bayes_data": keep_bayes_data,
            "fail_fast": fail_fast,
            "seed": seed,
        }
        for fold_id, heldout_id in enumerate(ids)
    ]

    results = execute_fold_calls(
        _prepared_bayesian_loio_fold,
        calls,
        client=client,
    )

    rows = [result["row"] for result in results]
    params = [row for result in results for row in result["params"]]
    curves = [
        result["boyce_bins"]
        for result in results
        if result["boyce_bins"] is not None and not result["boyce_bins"].empty
    ]
    diagnostics = {
        result["row"]["heldout_ID"]: result["diagnostics"] for result in results
    }

    return (
        pd.DataFrame(rows),
        pd.DataFrame(params),
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        diagnostics,
    )


__all__ = [
    "leave_one_individual_out_bayesian_rsf_prepared",
]
