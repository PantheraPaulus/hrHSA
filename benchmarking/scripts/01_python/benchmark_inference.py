"""Benchmark hrHSA statistical inference independently of raster I/O.

The spatial benchmark campaign measures extraction and prediction kernels. This
script measures frequentist design-matrix/logistic optimization and hierarchical
Bayesian preparation/NUTS inference. Bayesian timing explicitly synchronizes JAX
backends before an inference operation is considered complete.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from threadpoolctl import threadpool_limits

from hsa import FeatureSpec
from hsa.compute import (
    append_benchmark_record,
    benchmark_timer,
    make_benchmark_record,
    read_benchmark_records,
)
from hsa.features import build_design_matrix
from hsa.rsf.bayesian import (
    build_bayesian_rsf_model,
    evaluate_bayesian_rsf,
    prepare_bayesian_rsf_data,
)
from hsa.rsf.bayesian_hpc import _materialize_inference_tree


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    result = [int(item) for item in _csv_strings(value)]
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("integer lists must contain positive values")
    return result


def _synthetic_frame(
    *,
    rows: int,
    predictors: int,
    individuals: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    if rows <= 0 or predictors <= 0 or individuals <= 0:
        raise ValueError("rows, predictors and individuals must all be positive")

    rng = np.random.default_rng(seed)
    names = [f"x{i}" for i in range(predictors)]
    X = rng.normal(size=(rows, predictors)).astype(np.float64, copy=False)
    beta = np.linspace(0.55, -0.25, predictors, dtype=np.float64)

    id_idx = np.arange(rows, dtype=np.int64) % individuals
    rng.shuffle(id_idx)
    individual_effect = rng.normal(0.0, 0.35, size=individuals)
    eta = -2.0 + X @ beta + individual_effect[id_idx]
    used = rng.binomial(1, expit(eta)).astype(np.int8)

    data: dict[str, Any] = {name: X[:, i] for i, name in enumerate(names)}
    data["used"] = used
    data["individual-local-identifier"] = pd.Categorical.from_codes(
        id_idx,
        categories=[f"id-{i:03d}" for i in range(individuals)],
    )
    return pd.DataFrame(data), names


def _spec(names: list[str]) -> FeatureSpec:
    return FeatureSpec(
        linear=names,
        quadratic=names[:1],
        interactions=[(names[0], names[1])] if len(names) > 1 else [],
        add_const=True,
    )


def _frame_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(index=True, deep=True).sum())


def _optimizer_metadata(result) -> dict[str, Any]:
    retvals = getattr(result, "mle_retvals", None) or {}
    return {
        "converged": bool(retvals.get("converged", True)),
        "iterations": retvals.get("iterations"),
        "function_calls": retvals.get("fcalls"),
        "gradient_calls": retvals.get("gcalls"),
        "llf": float(result.llf),
    }


def _write_summary(path: Path, output_dir: Path) -> None:
    if not path.exists():
        return
    raw = read_benchmark_records(path)
    raw.to_csv(output_dir / f"{path.stem}_raw.csv", index=False)
    if raw.empty:
        return
    variant = raw.get("metadata.variant")
    if variant is None:
        variant = pd.Series(["default"] * len(raw), index=raw.index)
    work = raw.assign(variant=variant.fillna("default"))
    aggregations: dict[str, tuple[str, object]] = {
        "repeats": ("wall_seconds", "count"),
        "median_seconds": ("wall_seconds", "median"),
        "q25_seconds": ("wall_seconds", lambda x: x.quantile(0.25)),
        "q75_seconds": ("wall_seconds", lambda x: x.quantile(0.75)),
        "median_rows_s": ("throughput_rows_s", "median"),
        "max_peak_rss_mb": ("peak_rss_mb", "max"),
    }
    if "operation_peak_rss_mb" in work:
        aggregations["median_operation_peak_rss_mb"] = (
            "operation_peak_rss_mb",
            "median",
        )
    if "operation_peak_process_tree_rss_mb" in work:
        aggregations["median_operation_peak_process_tree_rss_mb"] = (
            "operation_peak_process_tree_rss_mb",
            "median",
        )
    summary = (
        work.groupby(["benchmark", "variant"], dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["benchmark", "median_seconds"])
    )
    summary.to_csv(output_dir / f"{path.stem}_summary.csv", index=False)
    print(summary.to_string(index=False))


def _run_frequentist(args) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "frequentist_inference.jsonl"
    if output.exists():
        output.unlink()

    df, names = _synthetic_frame(
        rows=args.rows,
        predictors=args.predictors,
        individuals=args.individuals,
        seed=args.seed,
    )
    spec = _spec(names)
    bytes_processed = _frame_bytes(df)
    methods = _csv_strings(args.methods)
    thread_counts = _csv_ints(args.blas_threads)

    for threads in thread_counts:
        for repeat in range(1, args.repeats + 1):
            with threadpool_limits(limits=threads):
                with benchmark_timer() as timer:
                    X, _, _ = build_design_matrix(
                        df,
                        spec,
                        scaler=None,
                        fit_scaler=True,
                        meta=None,
                    )
                append_benchmark_record(
                    make_benchmark_record(
                        "inference_frequentist_design_matrix",
                        timer["wall_seconds"],
                        rows=len(df),
                        workers=1,
                        threads_per_worker=threads,
                        bytes_processed=bytes_processed,
                        metadata={
                            "campaign": "inference_frequentist",
                            "variant": f"{threads}blas",
                            "stage": "design_matrix",
                            "blas_threads": threads,
                            "predictors_raw": len(names),
                            "design_columns": int(X.shape[1]),
                            "repeat": repeat,
                            "memory_monitor_interval_seconds": 0.25,
                        },
                        operation_memory=timer,
                    ),
                    output,
                )

                y = df.loc[X.index, "used"].astype(int)
                reference_params: np.ndarray | None = None
                for method in methods:
                    with benchmark_timer() as timer:
                        result = sm.Logit(y, X).fit(
                            method=method,
                            disp=False,
                            maxiter=args.maxiter,
                        )
                    params = np.asarray(result.params, dtype=float)
                    if reference_params is None:
                        reference_params = params.copy()
                    max_abs_diff = float(np.max(np.abs(params - reference_params)))
                    metadata = {
                        "campaign": "inference_frequentist",
                        "variant": f"{method}_{threads}blas",
                        "stage": "optimizer",
                        "optimizer": method,
                        "blas_threads": threads,
                        "predictors_raw": len(names),
                        "design_columns": int(X.shape[1]),
                        "max_abs_coef_diff_vs_first_method": max_abs_diff,
                        "repeat": repeat,
                        "memory_monitor_interval_seconds": 0.25,
                        **_optimizer_metadata(result),
                    }
                    append_benchmark_record(
                        make_benchmark_record(
                            "inference_frequentist_logit_fit",
                            timer["wall_seconds"],
                            rows=len(X),
                            workers=1,
                            threads_per_worker=threads,
                            bytes_processed=int(X.memory_usage(index=True, deep=True).sum()),
                            metadata=metadata,
                            operation_memory=timer,
                        ),
                        output,
                    )

    _write_summary(output, output_dir)


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
    candidates = []
    for group_name in ("sample_stats", "posterior"):
        group = getattr(idata, group_name, None)
        attrs = getattr(group, "attrs", {}) if group is not None else {}
        for name in ("sampling_time", "sampling_time_seconds"):
            if name in attrs:
                candidates.append(attrs[name])
    for value in candidates:
        try:
            return float(value)
        except Exception:
            continue
    return None


def _eta_modes(value: str) -> list[bool]:
    if value == "off":
        return [False]
    if value == "on":
        return [True]
    return [False, True]


def _failure_metadata(
    *,
    sampler: str,
    mode: str,
    repeat: int,
    exc: BaseException,
    stage: str,
    sampling_seed: int,
) -> dict[str, Any]:
    return {
        "campaign": "inference_bayesian",
        "variant": f"{sampler}_{mode}",
        "stage": stage,
        "status": "failed",
        "nuts_sampler": sampler,
        "repeat": repeat,
        "sampling_seed": sampling_seed,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc)[:1000],
    }


def _save_posterior_summary(idata, path: Path) -> None:
    import arviz as az

    summary = az.summary(idata, kind="stats", round_to=None)
    summary.index.name = "parameter"
    summary.to_csv(path)


def _run_bayesian(args) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "bayesian_inference.jsonl"
    if output.exists():
        output.unlink()

    try:
        import pymc as pm
    except ImportError as exc:
        raise SystemExit("Bayesian benchmark requires `pip install -e '.[bayesian]'`.") from exc

    df, names = _synthetic_frame(
        rows=args.rows,
        predictors=args.predictors,
        individuals=args.individuals,
        seed=args.seed,
    )
    binning = {name: args.bin_width for name in names}
    samplers = _csv_strings(args.samplers)
    sample_signature = inspect.signature(pm.sample)
    sampling_seed_base = args.seed if args.sampling_seed is None else args.sampling_seed

    for repeat in range(1, args.repeats + 1):
        with benchmark_timer() as timer:
            bayes_data = prepare_bayesian_rsf_data(
                df,
                names,
                id_col="individual-local-identifier",
                binning=binning,
            )
        model_meta = bayes_data["meta"]["_model"]
        append_benchmark_record(
            make_benchmark_record(
                "inference_bayesian_data_prepare",
                timer["wall_seconds"],
                rows=len(df),
                bytes_processed=_frame_bytes(df),
                metadata={
                    "campaign": "inference_bayesian",
                    "variant": "aggregation",
                    "stage": "data_prepare",
                    "n_raw": model_meta["n_raw"],
                    "n_aggregated": model_meta["n_aggregated"],
                    "compression_ratio": model_meta["compression_ratio"],
                    "predictors": len(names),
                    "individuals": args.individuals,
                    "bin_width": args.bin_width,
                    "repeat": repeat,
                    "memory_monitor_interval_seconds": 0.25,
                },
                operation_memory=timer,
            ),
            output,
        )

        for store_eta in _eta_modes(args.eta_storage):
            for sampler in samplers:
                with benchmark_timer() as timer:
                    model = build_bayesian_rsf_model(
                        bayes_data,
                        predictors=names,
                        random_intercept=True,
                        random_slopes=names[: args.random_slopes],
                        store_eta=store_eta,
                    )
                mode = "eta_on" if store_eta else "eta_off"
                append_benchmark_record(
                    make_benchmark_record(
                        "inference_bayesian_model_build",
                        timer["wall_seconds"],
                        rows=model_meta["n_aggregated"],
                        metadata={
                            "campaign": "inference_bayesian",
                            "variant": f"{sampler}_{mode}",
                            "stage": "model_build",
                            "nuts_sampler": sampler,
                            "store_eta": store_eta,
                            "n_aggregated": model_meta["n_aggregated"],
                            "repeat": repeat,
                            "memory_monitor_interval_seconds": 0.25,
                        },
                        operation_memory=timer,
                    ),
                    output,
                )

                sampling_seed = int(sampling_seed_base + repeat)
                sampling_kwargs: dict[str, Any] = {
                    "draws": args.draws,
                    "tune": args.tune,
                    "chains": args.chains,
                    "cores": args.cores,
                    "target_accept": args.target_accept,
                    "progressbar": False,
                    "return_inferencedata": True,
                    "random_seed": sampling_seed,
                }
                if "compute_convergence_checks" in sample_signature.parameters:
                    sampling_kwargs["compute_convergence_checks"] = False
                if "blas_cores" in sample_signature.parameters and args.blas_cores is not None:
                    sampling_kwargs["blas_cores"] = args.blas_cores
                if "nuts_sampler" in sample_signature.parameters:
                    sampling_kwargs["nuts_sampler"] = sampler
                elif sampler != "pymc":
                    exc = RuntimeError("this PyMC version has no nuts_sampler argument")
                    append_benchmark_record(
                        make_benchmark_record(
                            "inference_bayesian_failure",
                            0.0,
                            rows=model_meta["n_aggregated"],
                            metadata=_failure_metadata(
                                sampler=sampler,
                                mode=mode,
                                repeat=repeat,
                                exc=exc,
                                stage="sample",
                                sampling_seed=sampling_seed,
                            ),
                        ),
                        output,
                    )
                    print(f"Skipping {sampler}: {exc}")
                    continue

                try:
                    with benchmark_timer() as sample_timer:
                        dispatch_started = perf_counter()
                        with model:
                            raw_idata = pm.sample(**sampling_kwargs)
                        dispatch_seconds = perf_counter() - dispatch_started
                        materialize_started = perf_counter()
                        idata = _materialize_inference_tree(raw_idata)
                        materialize_seconds = perf_counter() - materialize_started
                    completed_sampling_seconds = sample_timer["wall_seconds"]
                except Exception as exc:
                    failed_seconds = sample_timer.get("wall_seconds", 0.0) if "sample_timer" in locals() else 0.0
                    append_benchmark_record(
                        make_benchmark_record(
                            "inference_bayesian_failure",
                            failed_seconds,
                            rows=model_meta["n_aggregated"],
                            workers=args.chains,
                            metadata={
                                **_failure_metadata(
                                    sampler=sampler,
                                    mode=mode,
                                    repeat=repeat,
                                    exc=exc,
                                    stage="sample_or_materialize",
                                    sampling_seed=sampling_seed,
                                ),
                                "n_raw": model_meta["n_raw"],
                                "n_aggregated": model_meta["n_aggregated"],
                            },
                        ),
                        output,
                    )
                    print(f"Failed {sampler}: {type(exc).__name__}: {exc}")
                    continue

                with benchmark_timer() as diagnostics_timer:
                    diagnostics = evaluate_bayesian_rsf(
                        idata,
                        include_random_effects=False,
                    )
                diag_table = diagnostics["diagnostics"]
                median_ess_bulk = (
                    float(diag_table["ess_bulk"].median())
                    if "ess_bulk" in diag_table
                    else np.nan
                )
                posterior_bytes = _dataset_nbytes(getattr(idata, "posterior", None))
                sampler_seconds = _sampler_reported_seconds(idata)
                summary_path = None
                if args.save_posterior_summary:
                    summary_path = output_dir / f"posterior_{sampler}_{mode}_repeat{repeat}.csv"
                    _save_posterior_summary(idata, summary_path)

                metadata = {
                    "campaign": "inference_bayesian",
                    "variant": f"{sampler}_{mode}",
                    "stage": "sample",
                    "status": "success",
                    "nuts_sampler": sampler,
                    "store_eta": store_eta,
                    "eta_present_in_posterior": "eta" in idata.posterior,
                    "draws": args.draws,
                    "tune": args.tune,
                    "chains": args.chains,
                    "cores": args.cores,
                    "blas_cores": args.blas_cores,
                    "target_accept": args.target_accept,
                    "sampling_seed": sampling_seed,
                    "n_raw": model_meta["n_raw"],
                    "n_aggregated": model_meta["n_aggregated"],
                    "compression_ratio": model_meta["compression_ratio"],
                    "posterior_bytes": posterior_bytes,
                    "posterior_mib": posterior_bytes / 1024**2,
                    "posterior_summary": str(summary_path) if summary_path is not None else None,
                    "sampler_reported_seconds": sampler_seconds,
                    "dispatch_seconds": dispatch_seconds,
                    "materialize_seconds": materialize_seconds,
                    "completed_sampling_seconds": completed_sampling_seconds,
                    "diagnostics_seconds": diagnostics_timer["wall_seconds"],
                    "raw_draws_per_second": args.draws * args.chains / completed_sampling_seconds,
                    "min_ess_bulk": diagnostics["min_ess_bulk"],
                    "median_ess_bulk": median_ess_bulk,
                    "min_ess_bulk_per_second": diagnostics["min_ess_bulk"]
                    / completed_sampling_seconds,
                    "median_ess_bulk_per_second": median_ess_bulk
                    / completed_sampling_seconds,
                    "min_ess_tail": diagnostics["min_ess_tail"],
                    "max_rhat": diagnostics["max_rhat"],
                    "n_divergences": diagnostics["n_divergences"],
                    "max_tree_depth": diagnostics["max_tree_depth"],
                    "mean_n_steps": diagnostics["mean_n_steps"],
                    "repeat": repeat,
                    "memory_monitor_interval_seconds": 0.25,
                }
                append_benchmark_record(
                    make_benchmark_record(
                        "inference_bayesian_nuts",
                        completed_sampling_seconds,
                        rows=model_meta["n_aggregated"],
                        workers=args.chains,
                        threads_per_worker=None,
                        metadata=metadata,
                        operation_memory=sample_timer,
                    ),
                    output,
                )

    _write_summary(output, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark frequentist and Bayesian hrHSA inference independently of raster I/O."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    frequentist = subparsers.add_parser("frequentist")
    frequentist.add_argument("--output-dir", type=Path, required=True)
    frequentist.add_argument("--rows", type=int, default=500_000)
    frequentist.add_argument("--predictors", type=int, default=6)
    frequentist.add_argument("--individuals", type=int, default=20)
    frequentist.add_argument("--methods", default="newton,lbfgs,bfgs")
    frequentist.add_argument("--blas-threads", default="1,2,4,8,12")
    frequentist.add_argument("--maxiter", type=int, default=100)
    frequentist.add_argument("--repeats", type=int, default=3)
    frequentist.add_argument("--seed", type=int, default=42)
    frequentist.set_defaults(func=_run_frequentist)

    bayesian = subparsers.add_parser("bayesian")
    bayesian.add_argument("--output-dir", type=Path, required=True)
    bayesian.add_argument("--rows", type=int, default=100_000)
    bayesian.add_argument("--predictors", type=int, default=3)
    bayesian.add_argument("--individuals", type=int, default=20)
    bayesian.add_argument("--bin-width", type=float, default=0.5)
    bayesian.add_argument("--random-slopes", type=int, default=1)
    bayesian.add_argument("--samplers", default="pymc,nutpie,numpyro,blackjax")
    bayesian.add_argument("--draws", type=int, default=500)
    bayesian.add_argument("--tune", type=int, default=500)
    bayesian.add_argument("--chains", type=int, default=4)
    bayesian.add_argument("--cores", type=int, default=4)
    bayesian.add_argument("--blas-cores", type=int, default=None)
    bayesian.add_argument("--target-accept", type=float, default=0.9)
    bayesian.add_argument("--eta-storage", choices=("off", "on", "both"), default="off")
    bayesian.add_argument("--repeats", type=int, default=1)
    bayesian.add_argument("--seed", type=int, default=42)
    bayesian.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Base sampling seed; synthetic data continue to use --seed.",
    )
    bayesian.add_argument(
        "--save-posterior-summary",
        action="store_true",
        help="Write compact ArviZ posterior mean/SD summaries for backend validation.",
    )
    bayesian.set_defaults(func=_run_bayesian)

    args = parser.parse_args()
    if getattr(args, "random_slopes", 0) > getattr(args, "predictors", 0):
        parser.error("--random-slopes cannot exceed --predictors")
    if getattr(args, "repeats", 1) <= 0:
        parser.error("--repeats must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
