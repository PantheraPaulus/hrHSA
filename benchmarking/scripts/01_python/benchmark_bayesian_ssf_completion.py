"""Fresh-process completed-inference benchmark for hierarchical Bayesian SSF/iSSF.

The workload preserves one categorical outcome per fixed-size choice stratum. iSSF
adds movement-style predictors and a centred proposal offset so the likelihood shape
matches the production integrated step-selection model more closely than a generic
categorical benchmark.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from benchmark_bayesian_completion import (
    _backing_types,
    _dataset_nbytes,
    _materialize_groups,
    _sampler_reported_seconds,
    _save_summary,
)
from hsa.compute import operation_memory_monitor
from hsa.ssf.bayesian import build_hierarchical_ssf_model
from hsa.ssf.data import build_ssf_choice_arrays


def _predictor_names(analysis: str, predictors: int) -> list[str]:
    if predictors <= 0:
        raise ValueError("predictors must be positive")
    if analysis == "ssf":
        return [f"x{i}" for i in range(predictors)]
    if predictors < 3:
        raise ValueError("iSSF benchmark requires at least three predictors")
    return [*[f"x{i}" for i in range(predictors - 2)], "log_sl", "cos_ta"]


def _synthetic_choices(
    *,
    n_strata: int,
    n_choices: int,
    predictors: int,
    individuals: int,
    analysis: str,
    seed: int,
) -> tuple[pd.DataFrame, list[str], str | None]:
    rng = np.random.default_rng(seed)
    names = _predictor_names(analysis, predictors)
    X = rng.normal(size=(n_strata, n_choices, predictors)).astype(np.float32)

    if analysis == "issf":
        X[:, :, names.index("log_sl")] = rng.normal(
            0.0, 0.7, size=(n_strata, n_choices)
        )
        X[:, :, names.index("cos_ta")] = rng.uniform(
            -1.0, 1.0, size=(n_strata, n_choices)
        )
        offset = rng.normal(0.0, 0.35, size=(n_strata, n_choices)).astype(np.float32)
        offset -= offset.mean(axis=1, keepdims=True)
        offset_col = "proposal_offset"
    else:
        offset = np.zeros((n_strata, n_choices), dtype=np.float32)
        offset_col = None

    beta = np.linspace(0.45, -0.20, predictors, dtype=np.float32)
    eta = np.einsum("sjp,p->sj", X, beta, optimize=True) + offset
    probability = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
    cumulative = np.cumsum(probability, axis=1)
    chosen = np.sum(rng.random(n_strata)[:, None] > cumulative, axis=1)
    chosen = np.minimum(chosen, n_choices - 1).astype(np.int32)

    strata = np.arange(n_strata, dtype=np.int64)
    id_by_stratum = strata % individuals
    used = np.zeros((n_strata, n_choices), dtype=np.int8)
    used[np.arange(n_strata), chosen] = 1

    data: dict[str, Any] = {
        "id": np.repeat(id_by_stratum, n_choices),
        "stratum_id": np.repeat(strata, n_choices),
        "candidate_id": np.tile(np.arange(n_choices, dtype=np.int16), n_strata),
        "used": used.reshape(-1),
    }
    flat = X.reshape(-1, predictors)
    for index, name in enumerate(names):
        data[name] = flat[:, index]
    if offset_col is not None:
        data[offset_col] = offset.reshape(-1)

    return pd.DataFrame(data), names, offset_col


def _diagnostics(idata, completed_seconds: float) -> dict[str, Any]:
    import arviz as az

    summary = az.summary(
        idata,
        var_names=["mu_beta", "sigma_beta"],
        round_to=None,
    )
    ess_bulk = summary["ess_bulk"].to_numpy(dtype=float)
    ess_tail = summary["ess_tail"].to_numpy(dtype=float)
    rhat = summary["r_hat"].to_numpy(dtype=float)

    sample_stats = getattr(idata, "sample_stats", None)
    divergences = 0
    mean_n_steps = np.nan
    max_tree_depth = np.nan
    if sample_stats is not None:
        if "diverging" in sample_stats:
            divergences = int(np.asarray(sample_stats["diverging"]).sum())
        elif "divergences" in sample_stats:
            divergences = int(np.asarray(sample_stats["divergences"]).sum())
        if "n_steps" in sample_stats:
            mean_n_steps = float(np.asarray(sample_stats["n_steps"]).mean())
        if "tree_depth" in sample_stats:
            max_tree_depth = int(np.asarray(sample_stats["tree_depth"]).max())

    return {
        "min_ess_bulk": float(np.nanmin(ess_bulk)),
        "median_ess_bulk": float(np.nanmedian(ess_bulk)),
        "min_ess_tail": float(np.nanmin(ess_tail)),
        "median_ess_tail": float(np.nanmedian(ess_tail)),
        "min_ess_bulk_per_completed_second": float(np.nanmin(ess_bulk))
        / completed_seconds,
        "median_ess_bulk_per_completed_second": float(np.nanmedian(ess_bulk))
        / completed_seconds,
        "min_ess_tail_per_completed_second": float(np.nanmin(ess_tail))
        / completed_seconds,
        "median_ess_tail_per_completed_second": float(np.nanmedian(ess_tail))
        / completed_seconds,
        "max_rhat": float(np.nanmax(rhat)),
        "n_divergences": divergences,
        "mean_n_steps": mean_n_steps,
        "max_tree_depth": max_tree_depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark completed Bayesian SSF/iSSF inference in one fresh process."
    )
    parser.add_argument("--analysis", choices=("ssf", "issf"), default="ssf")
    parser.add_argument("--sampler", default="blackjax")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--posterior-summary", type=Path, default=None)
    parser.add_argument("--strata", type=int, default=10_000)
    parser.add_argument("--choices", type=int, default=11)
    parser.add_argument("--predictors", type=int, default=6)
    parser.add_argument("--individuals", type=int, default=20)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--blas-cores", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=10001)
    parser.add_argument("--memory-sample-interval", type=float, default=0.25)
    args = parser.parse_args()

    if min(args.strata, args.choices, args.predictors, args.individuals) <= 0:
        parser.error("workload dimensions must be positive")
    if args.choices < 2:
        parser.error("--choices must be at least 2")
    if args.memory_sample_interval <= 0:
        parser.error("--memory-sample-interval must be positive")

    import pymc as pm

    prepare_start = time.perf_counter()
    frame, names, offset_col = _synthetic_choices(
        n_strata=args.strata,
        n_choices=args.choices,
        predictors=args.predictors,
        individuals=args.individuals,
        analysis=args.analysis,
        seed=args.data_seed,
    )
    arrays = build_ssf_choice_arrays(
        frame,
        id_col="id",
        predictors=names,
        stratum_col="stratum_id",
        candidate_col="candidate_id",
        used_col="used",
        offset_col=offset_col,
        dtype="float32",
    )
    prepare_seconds = time.perf_counter() - prepare_start

    model_start = time.perf_counter()
    model = build_hierarchical_ssf_model(arrays)
    model_seconds = time.perf_counter() - model_start

    signature = inspect.signature(pm.sample)
    sampling: dict[str, Any] = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "progressbar": False,
        "return_inferencedata": True,
        "random_seed": args.sampling_seed,
    }
    if "compute_convergence_checks" in signature.parameters:
        sampling["compute_convergence_checks"] = False
    if "nuts_sampler" in signature.parameters:
        sampling["nuts_sampler"] = args.sampler
    elif args.sampler != "pymc":
        raise RuntimeError("This PyMC version has no nuts_sampler argument")
    if "blas_cores" in signature.parameters and args.blas_cores is not None:
        sampling["blas_cores"] = args.blas_cores

    with operation_memory_monitor(interval_seconds=args.memory_sample_interval) as memory:
        dispatch_start = time.perf_counter()
        with model:
            raw_idata = pm.sample(**sampling)
        dispatch_seconds = time.perf_counter() - dispatch_start
        sampler_reported_seconds = _sampler_reported_seconds(raw_idata)
        before = {
            "posterior": _backing_types(getattr(raw_idata, "posterior", None)),
            "sample_stats": _backing_types(getattr(raw_idata, "sample_stats", None)),
        }
        materialize_start = time.perf_counter()
        idata = _materialize_groups(raw_idata)
        materialize_seconds = time.perf_counter() - materialize_start
        completed_seconds = dispatch_seconds + materialize_seconds

    diagnostics_start = time.perf_counter()
    diagnostics = _diagnostics(idata, completed_seconds)
    diagnostics_seconds = time.perf_counter() - diagnostics_start

    if args.posterior_summary is not None:
        args.posterior_summary.parent.mkdir(parents=True, exist_ok=True)
        _save_summary(idata, args.posterior_summary)

    posterior_bytes = _dataset_nbytes(getattr(idata, "posterior", None))
    result = {
        "status": "success",
        "analysis": args.analysis,
        "sampler": args.sampler,
        "prepare_seconds": prepare_seconds,
        "model_seconds": model_seconds,
        "dispatch_seconds": dispatch_seconds,
        "materialize_seconds": materialize_seconds,
        "completed_sampling_seconds": completed_seconds,
        "diagnostics_seconds": diagnostics_seconds,
        "completed_plus_diagnostics_seconds": completed_seconds + diagnostics_seconds,
        "sampler_reported_seconds": sampler_reported_seconds,
        "raw_draws_per_completed_second": args.draws * args.chains / completed_seconds,
        "posterior_bytes": posterior_bytes,
        "posterior_mib": posterior_bytes / 1024**2,
        "posterior_summary": (
            str(args.posterior_summary.resolve()) if args.posterior_summary is not None else None
        ),
        "operation_peak_rss_mb": memory["operation_peak_rss_mb"],
        "operation_peak_process_tree_rss_mb": memory[
            "operation_peak_process_tree_rss_mb"
        ],
        "memory_samples": memory["memory_samples"],
        "memory_sample_interval_seconds": args.memory_sample_interval,
        "backing_types_before": before,
        "n_strata": args.strata,
        "n_choices": args.choices,
        "choice_rows": int(len(frame)),
        "predictors": args.predictors,
        "individuals": args.individuals,
        "offset": offset_col is not None,
        "offset_sd": float(arrays.offset.std()) if arrays.offset is not None else 0.0,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "data_seed": args.data_seed,
        "sampling_seed": args.sampling_seed,
        **diagnostics,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
