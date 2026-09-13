"""Benchmark outer-model Bayesian RSF throughput at fixed statistical size.

Independent synthetic RSF posteriors are the distributed unit. Unlike the LOIO
CV scaling harness, increasing ``--models`` does not increase the number of
individuals or rows inside each fit. This makes the benchmark suitable for
isolating outer scheduling and sampler geometry on large CPU nodes.

Distributed callables intentionally depend only on installed hrHSA/package modules,
not sibling benchmark scripts. Dask schedulers and workers are commonly launched
with a different ``sys.path`` than the driver script on HPC systems.
"""

from __future__ import annotations

import argparse
import inspect
import os
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from hsa.compute import append_benchmark_record, benchmark_timer, make_benchmark_record
from hsa.rsf.bayesian import (
    build_bayesian_rsf_model,
    evaluate_bayesian_rsf,
    prepare_bayesian_rsf_data,
)
from hsa.rsf.bayesian_hpc import _materialize_inference_tree
from hsa.rsf.cv_parallel import execute_fold_calls


def _distribution_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _fit_rsf_model(
    *,
    rows: int,
    predictors: int,
    individuals: int,
    bin_width: float,
    random_slopes: int,
    sampler: str,
    draws: int,
    tune: int,
    chains: int,
    cores_per_fit: int,
    target_accept: float,
    data_seed: int,
    sampling_seed: int,
) -> dict[str, Any]:
    started = perf_counter()
    stage = "prepare"
    try:
        import pandas as pd
        import pymc as pm
        from dask.distributed import get_worker

        worker = get_worker()

        prepare_started = perf_counter()
        rng = np.random.default_rng(data_seed)
        names = [f"x{i}" for i in range(predictors)]
        X = rng.normal(size=(rows, predictors)).astype(np.float64, copy=False)
        beta = np.linspace(0.55, -0.25, predictors, dtype=np.float64)
        individual_index = np.arange(rows, dtype=np.int64) % individuals
        individual_effect = rng.normal(0.0, 0.35, size=individuals)
        eta = -2.0 + X @ beta + individual_effect[individual_index]
        probability = 1.0 / (1.0 + np.exp(-eta))
        used = rng.binomial(1, probability).astype(np.int8)
        if rows >= 2:
            used[0] = 1
            used[1] = 0

        frame_data: dict[str, Any] = {
            name: X[:, column] for column, name in enumerate(names)
        }
        frame_data["used"] = used
        frame_data["individual-local-identifier"] = np.array(
            [f"id-{value:04d}" for value in individual_index],
            dtype=object,
        )
        frame = pd.DataFrame(frame_data)

        data = prepare_bayesian_rsf_data(
            frame,
            names,
            id_col="individual-local-identifier",
            binning={name: bin_width for name in names},
        )
        prepare_seconds = perf_counter() - prepare_started

        stage = "build_model"
        model_started = perf_counter()
        model = build_bayesian_rsf_model(
            data,
            predictors=names,
            random_intercept=True,
            random_slopes=names[:random_slopes],
            store_eta=False,
        )
        model_seconds = perf_counter() - model_started

        sampling: dict[str, Any] = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "cores": cores_per_fit,
            "target_accept": target_accept,
            "progressbar": False,
            "return_inferencedata": True,
            "random_seed": sampling_seed,
        }
        signature = inspect.signature(pm.sample)
        if "compute_convergence_checks" in signature.parameters:
            sampling["compute_convergence_checks"] = False
        if "blas_cores" in signature.parameters:
            sampling["blas_cores"] = 1
        if "nuts_sampler" in signature.parameters:
            sampling["nuts_sampler"] = sampler
        elif sampler != "pymc":
            raise RuntimeError("This PyMC version has no nuts_sampler argument")

        stage = "sample"
        sample_started = perf_counter()
        with model:
            raw_idata = pm.sample(**sampling)
        dispatch_seconds = perf_counter() - sample_started

        stage = "materialize"
        materialize_started = perf_counter()
        idata = _materialize_inference_tree(raw_idata)
        materialize_seconds = perf_counter() - materialize_started
        completed_sampling_seconds = dispatch_seconds + materialize_seconds
        del raw_idata

        stage = "diagnostics"
        diagnostics_started = perf_counter()
        diagnostics = evaluate_bayesian_rsf(idata, include_random_effects=False)
        diagnostics_seconds = perf_counter() - diagnostics_started
        table = diagnostics["diagnostics"]
        median_ess_bulk = (
            float(table["ess_bulk"].median()) if "ess_bulk" in table else np.nan
        )
        model_meta = data["meta"]["_model"]

        return {
            "status": "success",
            "wall_seconds": perf_counter() - started,
            "prepare_seconds": prepare_seconds,
            "model_seconds": model_seconds,
            "dispatch_seconds": dispatch_seconds,
            "materialize_seconds": materialize_seconds,
            "completed_sampling_seconds": completed_sampling_seconds,
            "diagnostics_seconds": diagnostics_seconds,
            "rows_raw": int(model_meta["n_raw"]),
            "rows_aggregated": int(model_meta["n_aggregated"]),
            "compression_ratio": float(model_meta["compression_ratio"]),
            "min_ess_bulk": float(diagnostics["min_ess_bulk"]),
            "median_ess_bulk": median_ess_bulk,
            "min_ess_tail": float(diagnostics["min_ess_tail"]),
            "max_rhat": float(diagnostics["max_rhat"]),
            "n_divergences": int(diagnostics["n_divergences"]),
            "mean_n_steps": float(diagnostics["mean_n_steps"]),
            "worker_address": worker.address,
            "worker_name": str(worker.name),
            "worker_pid": os.getpid(),
            "pymc_version": _distribution_version("pymc"),
            "nutpie_version": _distribution_version("nutpie"),
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "stage": stage,
            "wall_seconds": perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _median(results: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(result[key])
        for result in results
        if result.get("status") == "success" and result.get(key) is not None
    ]
    return float(np.median(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed-size outer-parallel Bayesian RSF fits."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--cores-per-fit", type=int, required=True)
    parser.add_argument("--models", type=int, default=112)
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--predictors", type=int, default=3)
    parser.add_argument("--individuals", type=int, default=24)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--random-slopes", type=int, default=1)
    parser.add_argument("--sampler", default="nutpie")
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--tune", type=int, default=250)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=10_000)
    parser.add_argument("--memory-monitor-interval", type=float, default=5.0)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    args = parser.parse_args()

    for name in (
        "workers", "cores_per_fit", "models", "rows", "predictors",
        "individuals", "draws", "tune", "chains",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.random_slopes < 0 or args.random_slopes > args.predictors:
        parser.error("--random-slopes must be between 0 and --predictors")
    if args.models < args.workers:
        parser.error("--models must be >= --workers")
    if args.memory_monitor_interval <= 0:
        parser.error("--memory-monitor-interval must be positive")

    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from dask.distributed import Client, LocalCluster

    cluster = None
    if args.scheduler_file is None:
        cluster = LocalCluster(
            n_workers=args.workers,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
        )
        client = Client(cluster)
    else:
        client = Client(scheduler_file=str(args.scheduler_file))

    try:
        client.wait_for_workers(args.workers, timeout=args.worker_startup_timeout)
        observed = len(client.scheduler_info().get("workers", {}))
        if observed != args.workers:
            raise RuntimeError(f"Expected {args.workers} workers; observed {observed}.")

        calls = [
            {
                "rows": args.rows,
                "predictors": args.predictors,
                "individuals": args.individuals,
                "bin_width": args.bin_width,
                "random_slopes": args.random_slopes,
                "sampler": args.sampler,
                "draws": args.draws,
                "tune": args.tune,
                "chains": args.chains,
                "cores_per_fit": args.cores_per_fit,
                "target_accept": args.target_accept,
                "data_seed": args.data_seed + 100_000 * index,
                "sampling_seed": args.sampling_seed + index,
            }
            for index in range(args.models)
        ]

        with benchmark_timer(
            client=client,
            memory_interval_seconds=args.memory_monitor_interval,
        ) as timer:
            results = execute_fold_calls(_fit_rsf_model, calls, client=client)

        successes = [r for r in results if r.get("status") == "success"]
        failures = [r for r in results if r.get("status") != "success"]
        completed = len(successes)
        wall = float(timer["wall_seconds"])
        metadata = {
            "campaign": "bayesian_rsf_outer_fixed_size_v1",
            "models_requested": args.models,
            "models_completed": completed,
            "models_failed": len(failures),
            "models_per_second": completed / wall if wall > 0 else np.nan,
            "workers": args.workers,
            "cores_per_fit": args.cores_per_fit,
            "rows_per_model": args.rows,
            "individuals_per_model": args.individuals,
            "predictors": args.predictors,
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "sampler": args.sampler,
            "memory_monitor_interval_seconds": args.memory_monitor_interval,
            "median_fit_seconds": _median(results, "wall_seconds"),
            "median_completed_sampling_seconds": _median(results, "completed_sampling_seconds"),
            "median_rows_aggregated": _median(results, "rows_aggregated"),
            "median_compression_ratio": _median(results, "compression_ratio"),
            "median_min_ess_bulk": _median(results, "min_ess_bulk"),
            "median_max_rhat": _median(results, "max_rhat"),
            "total_divergences": int(sum(int(r.get("n_divergences", 0)) for r in successes)),
            "failure_examples": [r.get("error") for r in failures[:3]],
        }
        record = make_benchmark_record(
            "bayesian_rsf_outer_fixed_size",
            wall,
            rows=args.models,
            workers=args.workers,
            threads_per_worker=1,
            bytes_processed=None,
            metadata=metadata,
            operation_memory=timer,
            client=client,
        )
        append_benchmark_record(record, args.output)
        print(
            f"rsf-fixed: workers={args.workers} cores/fit={args.cores_per_fit} "
            f"completed={completed}/{args.models} wall={wall:.3f}s "
            f"models/s={completed / wall if wall > 0 else np.nan:.4f}"
        )
        if failures:
            print("  failures:", metadata["failure_examples"])
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
