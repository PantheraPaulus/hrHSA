"""Benchmark completed Bayesian sampling across synchronous and asynchronous backends.

JAX-backed PyMC samplers may return device-backed arrays before all dispatched work
has completed. Timing only ``pm.sample()`` can therefore measure dispatch latency
rather than completed inference. This harness separates dispatch, materialization
and diagnostics, and measures operation-local memory across the complete sampling
plus materialization interval.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from benchmark_inference import _synthetic_frame
from hsa.compute import operation_memory_monitor
from hsa.rsf.bayesian import (
    build_bayesian_rsf_model,
    evaluate_bayesian_rsf,
    prepare_bayesian_rsf_data,
)


def _backing_types(dataset) -> dict[str, str]:
    if dataset is None:
        return {}
    return {
        name: f"{type(value.data).__module__}.{type(value.data).__name__}"
        for name, value in dataset.data_vars.items()
    }


def _numpy_dataset(dataset):
    """Return an equivalent Dataset whose data variables are host NumPy arrays."""
    if dataset is None:
        return None
    out = dataset.copy(deep=False)
    for name, value in dataset.data_vars.items():
        out[name] = value.copy(data=np.asarray(value.data))
    return out


def _materialize_groups(idata):
    """Materialize posterior/sample_stats and return a small DataTree."""
    groups: dict[str, Any] = {}
    for group_name in ("posterior", "sample_stats"):
        group = getattr(idata, group_name, None)
        if group is not None:
            groups[group_name] = _numpy_dataset(group)
    return xr.DataTree.from_dict(groups)


def _dataset_nbytes(dataset) -> int:
    if dataset is None:
        return 0
    total = 0
    for value in dataset.data_vars.values():
        try:
            total += int(np.prod(value.shape, dtype=np.int64)) * int(value.dtype.itemsize)
        except Exception:
            pass
    return total


def _sampler_reported_seconds(idata) -> float | None:
    for group_name in ("sample_stats", "posterior"):
        group = getattr(idata, group_name, None)
        attrs = getattr(group, "attrs", {}) if group is not None else {}
        for name in ("sampling_time", "sampling_time_seconds"):
            if name in attrs:
                try:
                    return float(attrs[name])
                except Exception:
                    pass
    return None


def _save_summary(normalized, path: Path) -> None:
    import arviz as az

    summary = az.summary(normalized, kind="stats", round_to=None)
    summary.index.name = "parameter"
    summary.to_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark completed Bayesian inference with explicit backend synchronization."
    )
    parser.add_argument("--sampler", default="blackjax")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--posterior-summary", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--predictors", type=int, default=3)
    parser.add_argument("--individuals", type=int, default=20)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--random-slopes", type=int, default=1)
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

    if args.random_slopes > args.predictors:
        parser.error("--random-slopes cannot exceed --predictors")
    if args.memory_sample_interval <= 0:
        parser.error("--memory-sample-interval must be positive")

    import pymc as pm

    frame, names = _synthetic_frame(
        rows=args.rows,
        predictors=args.predictors,
        individuals=args.individuals,
        seed=args.data_seed,
    )
    data = prepare_bayesian_rsf_data(
        frame,
        names,
        id_col="individual-local-identifier",
        binning={name: args.bin_width for name in names},
    )
    model = build_bayesian_rsf_model(
        data,
        predictors=names,
        random_intercept=True,
        random_slopes=names[: args.random_slopes],
        store_eta=False,
    )

    sample_signature = inspect.signature(pm.sample)
    sampling_kwargs: dict[str, Any] = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "progressbar": False,
        "return_inferencedata": True,
        "random_seed": args.sampling_seed,
    }
    if "compute_convergence_checks" in sample_signature.parameters:
        sampling_kwargs["compute_convergence_checks"] = False
    if "nuts_sampler" in sample_signature.parameters:
        sampling_kwargs["nuts_sampler"] = args.sampler
    elif args.sampler != "pymc":
        raise RuntimeError("This PyMC version has no nuts_sampler argument")
    if "blas_cores" in sample_signature.parameters and args.blas_cores is not None:
        sampling_kwargs["blas_cores"] = args.blas_cores

    # Keep the memory monitor alive until host materialization has completed. For
    # asynchronous JAX backends that is the actual completed-inference boundary.
    with operation_memory_monitor(
        interval_seconds=args.memory_sample_interval
    ) as memory:
        dispatch_start = time.perf_counter()
        with model:
            idata = pm.sample(**sampling_kwargs)
        dispatch_seconds = time.perf_counter() - dispatch_start

        sampler_reported_seconds = _sampler_reported_seconds(idata)
        before = {
            "posterior": _backing_types(getattr(idata, "posterior", None)),
            "sample_stats": _backing_types(getattr(idata, "sample_stats", None)),
        }

        materialize_start = time.perf_counter()
        normalized = _materialize_groups(idata)
        materialize_seconds = time.perf_counter() - materialize_start
        completed_sampling_seconds = dispatch_seconds + materialize_seconds

    after = {
        "posterior": _backing_types(getattr(normalized, "posterior", None)),
        "sample_stats": _backing_types(getattr(normalized, "sample_stats", None)),
    }

    diagnostics_start = time.perf_counter()
    diagnostics = evaluate_bayesian_rsf(normalized, include_random_effects=False)
    diagnostics_seconds = time.perf_counter() - diagnostics_start
    diag_table = diagnostics["diagnostics"]
    median_ess_bulk = (
        float(diag_table["ess_bulk"].median()) if "ess_bulk" in diag_table else np.nan
    )

    posterior_bytes = _dataset_nbytes(getattr(normalized, "posterior", None))
    if args.posterior_summary is not None:
        args.posterior_summary.parent.mkdir(parents=True, exist_ok=True)
        _save_summary(normalized, args.posterior_summary)

    result = {
        "sampler": args.sampler,
        "status": "success",
        "dispatch_seconds": dispatch_seconds,
        "materialize_seconds": materialize_seconds,
        "completed_sampling_seconds": completed_sampling_seconds,
        "diagnostics_seconds": diagnostics_seconds,
        "completed_plus_diagnostics_seconds": completed_sampling_seconds + diagnostics_seconds,
        "sampler_reported_seconds": sampler_reported_seconds,
        "raw_draws_per_completed_second": args.draws * args.chains / completed_sampling_seconds,
        "min_ess_bulk": diagnostics["min_ess_bulk"],
        "median_ess_bulk": median_ess_bulk,
        "min_ess_bulk_per_completed_second": diagnostics["min_ess_bulk"]
        / completed_sampling_seconds,
        "median_ess_bulk_per_completed_second": median_ess_bulk
        / completed_sampling_seconds,
        "min_ess_tail": diagnostics["min_ess_tail"],
        "max_rhat": diagnostics["max_rhat"],
        "n_divergences": diagnostics["n_divergences"],
        "max_tree_depth": diagnostics["max_tree_depth"],
        "mean_n_steps": diagnostics["mean_n_steps"],
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
        "backing_types_after": after,
        "rows_raw": args.rows,
        "rows_aggregated": int(data["meta"]["_model"]["n_aggregated"]),
        "compression_ratio": float(data["meta"]["_model"]["compression_ratio"]),
        "predictors": args.predictors,
        "individuals": args.individuals,
        "random_slopes": args.random_slopes,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
        "sampling_seed": args.sampling_seed,
        "data_seed": args.data_seed,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
