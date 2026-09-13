"""Benchmark nutpie NUTS for hierarchical Bayesian RSF outer-fold concurrency.

This is a benchmark-only sampler comparison.  It keeps the existing hrHSA
PyMC-built hierarchical RSF target and compares a CPU-native nutpie NUTS path
against the already-established PyMC/BlackJAX results.  ``cores_per_fold`` is
passed directly to nutpie through ``pm.sample(..., cores=...)``, so unlike the
BlackJAX/JAX path the intended inner chain concurrency is explicit.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as package_version
import inspect
import os
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any

import numpy as np

import benchmark_cv_scaling as cvbench
from hsa.compute import append_benchmark_record, benchmark_timer, make_benchmark_record
from hsa.compute.workloads import parse_positive_ints
from hsa.rsf.bayesian import build_bayesian_rsf_model, prepare_bayesian_rsf_data
from hsa.rsf.bayesian_cv import _sampling_diagnostics
from hsa.rsf.bayesian_hpc import _materialize_inference_tree
from hsa.rsf.cv_parallel import execute_fold_calls


def _distribution_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _sample_stat_mean(idata, names: tuple[str, ...]) -> float | None:
    stats = getattr(idata, "sample_stats", None)
    if stats is None:
        return None
    for name in names:
        if name in stats:
            values = np.asarray(stats[name], dtype=float)
            if values.size:
                return float(np.nanmean(values))
    return None


def _nutpie_fold(
    *,
    prepared_root: str,
    heldout_id: Any,
    predictors: list[str],
    bin_width: float,
    random_slopes: int,
    draws: int,
    tune: int,
    chains: int,
    cores_per_fold: int,
    target_accept: float,
    seed: int,
) -> dict[str, Any]:
    started = perf_counter()
    stage = "worker_setup"
    worker_address = None
    worker_name = None

    try:
        import pymc as pm
        from dask.distributed import get_worker

        try:
            import nutpie  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "nutpie is required for this benchmark; PyMC 6.2 requires nutpie>=0.16.10"
            ) from exc

        worker = get_worker()
        worker_address = worker.address
        worker_name = str(worker.name)

        stage = "load_prepared"
        prepared = cvbench.PreparedDataset.open(prepared_root)
        train_df = prepared.load(exclude=heldout_id)
        load_seconds = perf_counter() - started

        stage = "prepare_bayesian_data"
        prepare_started = perf_counter()
        bayes_data = prepare_bayesian_rsf_data(
            train_df,
            predictors,
            id_col="individual-local-identifier",
            binning={name: bin_width for name in predictors},
        )
        prepare_seconds = perf_counter() - prepare_started

        stage = "build_model"
        model_started = perf_counter()
        model = build_bayesian_rsf_model(
            bayes_data,
            predictors=predictors,
            random_intercept=True,
            random_slopes=predictors[:random_slopes],
            store_eta=False,
        )
        model_seconds = perf_counter() - model_started

        sampling: dict[str, Any] = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "cores": cores_per_fold,
            "target_accept": target_accept,
            "progressbar": False,
            "return_inferencedata": True,
            "random_seed": seed,
            "nuts_sampler": "nutpie",
        }
        signature = inspect.signature(pm.sample)
        if "compute_convergence_checks" in signature.parameters:
            sampling["compute_convergence_checks"] = False
        if "blas_cores" in signature.parameters:
            sampling["blas_cores"] = 1

        stage = "sample_dispatch"
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
        diagnostics = _sampling_diagnostics(idata)
        diagnostics_seconds = perf_counter() - diagnostics_started
        model_meta = bayes_data["meta"]["_model"]

        return {
            "heldout_ID": heldout_id,
            "status": "success",
            "stage": "complete",
            "wall_seconds": perf_counter() - started,
            "load_seconds": load_seconds,
            "prepare_seconds": prepare_seconds,
            "model_seconds": model_seconds,
            "dispatch_seconds": dispatch_seconds,
            "materialize_seconds": materialize_seconds,
            "completed_sampling_seconds": completed_sampling_seconds,
            "diagnostics_seconds": diagnostics_seconds,
            "mean_n_steps": _sample_stat_mean(idata, ("n_steps", "num_steps")),
            "mean_acceptance_rate": _sample_stat_mean(
                idata, ("acceptance_rate", "acceptance_probability", "accept_stat")
            ),
            "n_raw": int(model_meta["n_raw"]),
            "n_aggregated": int(model_meta["n_aggregated"]),
            "worker_address": worker_address,
            "worker_name": worker_name,
            "worker_pid": os.getpid(),
            "pymc_version": _distribution_version("pymc"),
            "nutpie_version": _distribution_version("nutpie"),
            **diagnostics,
            "error": None,
            "traceback": None,
        }
    except Exception as exc:
        return {
            "heldout_ID": heldout_id,
            "status": "failed",
            "stage": stage,
            "wall_seconds": perf_counter() - started,
            "worker_address": worker_address,
            "worker_name": worker_name,
            "worker_pid": os.getpid(),
            "pymc_version": _distribution_version("pymc"),
            "nutpie_version": _distribution_version("nutpie"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _median(results: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(result[key])
        for result in results
        if result.get("status") == "success" and result.get(key) is not None
    ]
    return float(np.median(values)) if values else None


def _run_point(args, *, workers: int, cores_per_fold: int, prepared, output: Path) -> None:
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    )
    client = Client(cluster)
    try:
        client.wait_for_workers(workers, timeout=args.worker_startup_timeout)
        observed = len(client.scheduler_info().get("workers", {}))
        if observed != workers:
            raise RuntimeError(f"Expected {workers} workers; observed {observed}")

        ids = prepared.individuals[: args.folds]
        names = list(prepared.predictors)

        for repeat in range(1, args.repeats + 1):
            calls = [
                {
                    "prepared_root": str(prepared.root),
                    "heldout_id": heldout_id,
                    "predictors": names,
                    "bin_width": args.bin_width,
                    "random_slopes": args.random_slopes,
                    "draws": args.draws,
                    "tune": args.tune,
                    "chains": args.chains,
                    "cores_per_fold": cores_per_fold,
                    "target_accept": args.target_accept,
                    "seed": args.seed + repeat + 100_000 * fold_index,
                }
                for fold_index, heldout_id in enumerate(ids)
            ]

            with benchmark_timer(client=client) as timer:
                results = execute_fold_calls(_nutpie_fold, calls, client=client)

            successes = [r for r in results if r.get("status") == "success"]
            failures = [r for r in results if r.get("status") != "success"]
            completed = len(successes)
            wall = float(timer["wall_seconds"])
            metadata = {
                "campaign": "bayesian-rsf-nutpie-v1",
                "sampler": "nutpie",
                "workers": workers,
                "cores_per_fold": cores_per_fold,
                "nominal_inner_chain_capacity": workers * cores_per_fold,
                "repeat": repeat,
                "folds_requested": len(ids),
                "folds_completed": completed,
                "folds_failed": len(failures),
                "folds_per_second": completed / wall if wall > 0 else np.nan,
                "chains": args.chains,
                "draws": args.draws,
                "tune": args.tune,
                "target_accept": args.target_accept,
                "median_fold_seconds": _median(results, "wall_seconds"),
                "median_completed_sampling_seconds": _median(
                    results, "completed_sampling_seconds"
                ),
                "median_dispatch_seconds": _median(results, "dispatch_seconds"),
                "median_materialize_seconds": _median(results, "materialize_seconds"),
                "median_mean_n_steps": _median(results, "mean_n_steps"),
                "median_mean_acceptance_rate": _median(results, "mean_acceptance_rate"),
                "median_min_ess_bulk": _median(results, "min_ess_bulk"),
                "median_min_ess_tail": _median(results, "min_ess_tail"),
                "median_max_rhat": _median(results, "max_rhat"),
                "total_divergences": int(
                    sum(int(r.get("n_divergences", 0)) for r in successes)
                ),
                "rows_per_individual": args.rows_per_individual,
                "predictors": args.predictors,
                "bin_width": args.bin_width,
                "random_slopes": args.random_slopes,
                "pymc_versions": sorted(
                    {str(r["pymc_version"]) for r in results if r.get("pymc_version")}
                ),
                "nutpie_versions": sorted(
                    {str(r["nutpie_version"]) for r in results if r.get("nutpie_version")}
                ),
                "failure_examples": [
                    f"{r.get('stage')}: {r.get('error')}" for r in failures[:3]
                ],
                "observed_workers": observed,
            }

            append_benchmark_record(
                make_benchmark_record(
                    "bayesian_rsf_nutpie",
                    wall,
                    rows=len(ids),
                    workers=workers,
                    threads_per_worker=1,
                    bytes_processed=int(prepared.manifest["n_rows"].sum()),
                    metadata=metadata,
                    operation_memory=timer,
                    client=client,
                ),
                output,
            )

            status = "ok" if not failures else ("failed" if not successes else "partial")
            print(
                f"nutpie: workers={workers} cores/fold={cores_per_fold} "
                f"status={status} completed={completed}/{len(ids)} "
                f"wall={wall:.3f}s fold={metadata['median_fold_seconds']} "
                f"ESS={metadata['median_min_ess_bulk']} "
                f"Rhat={metadata['median_max_rhat']} "
                f"div={metadata['total_divergences']}"
            )
            if failures:
                print("  failures:", metadata["failure_examples"])
    finally:
        client.close()
        cluster.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark nutpie NUTS on hierarchical Bayesian RSF outer folds."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=parse_positive_ints, default=[6])
    parser.add_argument("--cores-per-fold", type=parse_positive_ints, default=[4])
    parser.add_argument("--folds", type=int, default=24)
    parser.add_argument("--rows-per-individual", type=int, default=500)
    parser.add_argument("--predictors", type=int, default=3)
    parser.add_argument("--random-slopes", type=int, default=1)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--tune", type=int, default=250)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    args = parser.parse_args()

    for name in ("folds", "rows_per_individual", "predictors", "draws", "tune", "chains", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.random_slopes < 0 or args.random_slopes > args.predictors:
        parser.error("--random-slopes must be between 0 and --predictors")

    args.output = args.output.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cache = args.work_dir / (
        f"cv-cache-{args.folds}x{args.rows_per_individual}-"
        f"{args.predictors}p-seed{args.seed}"
    )
    prepared = cvbench._prepare_synthetic_cache(
        cache,
        folds=args.folds,
        rows_per_individual=args.rows_per_individual,
        predictors=args.predictors,
        seed=args.seed,
        overwrite=args.overwrite_cache,
    )

    for workers in args.workers:
        for cores_per_fold in args.cores_per_fold:
            _run_point(
                args,
                workers=int(workers),
                cores_per_fold=int(cores_per_fold),
                prepared=prepared,
                output=args.output,
            )


if __name__ == "__main__":
    main()
