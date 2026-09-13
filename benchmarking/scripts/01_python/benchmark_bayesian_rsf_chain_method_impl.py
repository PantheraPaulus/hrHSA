"""Corrected BlackJAX chain-method contention diagnostic.

The diagnostic deliberately avoids querying JAX devices before PyMC's JAX
sampling module has initialized. PyMC configures virtual CPU devices during
``pymc.sampling.jax`` import; calling ``jax.local_device_count()`` earlier can
freeze the backend at one device and manufacture a multi-chain ``pmap`` failure.

``chain_method='default'`` reproduces the historical
``pm.sample(..., nuts_sampler='blackjax')`` path. Explicit ``parallel`` and
``vectorized`` modes call PyMC's JAX driver directly, which also makes them
usable on PyMC 6.0--6.2 where ``pm.sample`` does not route ``chain_method``.
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


_WORKER_FOLD_SEQUENCE = 0


def _distribution_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _sample(
    pm,
    model,
    *,
    sampler: str,
    chain_method: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    seed: int,
):
    """Sample without initializing JAX before PyMC has configured its backend."""
    if chain_method == "default":
        sampling: dict[str, Any] = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "cores": cores,
            "target_accept": target_accept,
            "progressbar": False,
            "return_inferencedata": True,
            "random_seed": seed,
        }
        signature = inspect.signature(pm.sample)
        if "nuts_sampler" in signature.parameters:
            sampling["nuts_sampler"] = sampler
        elif sampler != "pymc":
            raise RuntimeError("this PyMC version has no nuts_sampler argument")
        if "compute_convergence_checks" in signature.parameters:
            sampling["compute_convergence_checks"] = False
        if "blas_cores" in signature.parameters:
            sampling["blas_cores"] = 1
        with model:
            return pm.sample(**sampling), "historical_default", None

    # Importing pymc.sampling.jax is intentionally the first operation that can
    # configure/initialize the JAX backend in this worker. Do not query devices
    # before this import.
    from pymc.sampling.jax import sample_jax_nuts

    with model:
        idata = sample_jax_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            model=model,
            progressbar=False,
            quiet=True,
            chain_method=chain_method,
            nuts_sampler=sampler,
            compute_convergence_checks=False,
        )
    return idata, "direct_jax_driver", {"chain_method": chain_method}


def _post_sampling_jax_runtime() -> tuple[int | None, list[str] | None]:
    """Inspect devices only after sampling has fixed the JAX backend geometry."""
    try:
        import jax

        return int(jax.local_device_count()), [str(device) for device in jax.local_devices()]
    except Exception as exc:  # pragma: no cover - diagnostic only
        return None, [f"{type(exc).__name__}: {exc}"]


def _diagnostic_fold(
    *,
    prepared_root: str,
    heldout_id: Any,
    predictors: list[str],
    bin_width: float,
    random_slopes: int,
    sampler: str,
    chain_method: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    seed: int,
) -> dict[str, Any]:
    global _WORKER_FOLD_SEQUENCE
    _WORKER_FOLD_SEQUENCE += 1
    sequence = _WORKER_FOLD_SEQUENCE
    started = perf_counter()

    worker_address = None
    worker_name = None
    sampling_configuration = None
    sampling_nuts_options = None
    stage = "worker_setup"
    pymc_version = _distribution_version("pymc")
    jax_version = _distribution_version("jax")
    blackjax_version = _distribution_version("blackjax")

    try:
        import pymc as pm
        from dask.distributed import get_worker

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

        stage = "sample_dispatch"
        sample_started = perf_counter()
        raw_idata, sampling_configuration, sampling_nuts_options = _sample(
            pm,
            model,
            sampler=sampler,
            chain_method=chain_method,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            seed=seed,
        )
        dispatch_seconds = perf_counter() - sample_started

        stage = "materialize"
        materialize_started = perf_counter()
        idata = _materialize_inference_tree(raw_idata)
        materialize_seconds = perf_counter() - materialize_started
        completed_sampling_seconds = dispatch_seconds + materialize_seconds
        del raw_idata

        # Crucially this is after sampling, not before.
        jax_local_device_count, jax_devices = _post_sampling_jax_runtime()

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
            "n_raw": int(model_meta["n_raw"]),
            "n_aggregated": int(model_meta["n_aggregated"]),
            "worker_address": worker_address,
            "worker_name": worker_name,
            "worker_pid": os.getpid(),
            "worker_fold_sequence": sequence,
            "pymc_version": pymc_version,
            "jax_version": jax_version,
            "blackjax_version": blackjax_version,
            "sampling_configuration": sampling_configuration,
            "sampling_nuts_options": sampling_nuts_options,
            "jax_device_probe_timing": "post_sampling",
            "jax_local_device_count": jax_local_device_count,
            "jax_devices": jax_devices,
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
            "worker_fold_sequence": sequence,
            "pymc_version": pymc_version,
            "jax_version": jax_version,
            "blackjax_version": blackjax_version,
            "sampling_configuration": sampling_configuration,
            "sampling_nuts_options": sampling_nuts_options,
            "jax_device_probe_timing": "not_queried_before_failure",
            "jax_local_device_count": None,
            "jax_devices": None,
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


def _sequence_medians(results: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    sequences = sorted(
        {
            int(result["worker_fold_sequence"])
            for result in results
            if result.get("status") == "success"
        }
    )
    for sequence in sequences:
        values = [
            float(result["completed_sampling_seconds"])
            for result in results
            if result.get("status") == "success"
            and int(result["worker_fold_sequence"]) == sequence
        ]
        if values:
            output[f"sequence_{sequence}"] = float(np.median(values))
    return output


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "NA"
    value = float(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3f}s"


def _distinct_failure_examples(
    failures: list[dict[str, Any]], *, limit: int = 3
) -> list[str]:
    examples: list[str] = []
    seen: set[str] = set()
    for result in failures:
        qualified = (
            f"{result.get('stage') or 'unknown_stage'}: "
            f"{result.get('error') or 'unknown error'}"
        )
        if qualified in seen:
            continue
        seen.add(qualified)
        examples.append(qualified)
        if len(examples) >= limit:
            break
    return examples


def _runtime_values(results: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({str(result[key]) for result in results if result.get(key) is not None})


def _run_point(args, *, workers: int, chain_method: str, prepared, output: Path) -> None:
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
                    "sampler": args.sampler,
                    "chain_method": chain_method,
                    "draws": args.draws,
                    "tune": args.tune,
                    "chains": args.chains,
                    "cores": args.cores,
                    "target_accept": args.target_accept,
                    "seed": args.seed + repeat + 100_000 * fold_id,
                }
                for fold_id, heldout_id in enumerate(ids)
            ]

            with benchmark_timer(client=client) as timer:
                results = execute_fold_calls(_diagnostic_fold, calls, client=client)

            successes = [r for r in results if r.get("status") == "success"]
            failures = [r for r in results if r.get("status") != "success"]
            completed = len(successes)
            wall = float(timer["wall_seconds"])
            failure_examples = _distinct_failure_examples(failures)

            fold_records = [
                {
                    key: result.get(key)
                    for key in (
                        "heldout_ID", "status", "stage", "worker_address",
                        "worker_name", "worker_pid", "worker_fold_sequence",
                        "wall_seconds", "dispatch_seconds", "materialize_seconds",
                        "completed_sampling_seconds", "pymc_version", "jax_version",
                        "blackjax_version", "sampling_configuration",
                        "sampling_nuts_options", "jax_device_probe_timing",
                        "jax_local_device_count", "jax_devices", "error", "traceback",
                    )
                }
                for result in results
            ]
            metadata = {
                "campaign": "bayesian-rsf-chain-method-v3",
                "sampler": args.sampler,
                "chain_method": chain_method,
                "workers": workers,
                "repeat": repeat,
                "folds_requested": len(ids),
                "folds_completed": completed,
                "folds_failed": len(failures),
                "folds_per_second": completed / wall if wall > 0 else np.nan,
                "allocated_core_seconds_per_completed_fold": (
                    wall * workers / completed if completed else np.nan
                ),
                "median_fold_seconds": _median(results, "wall_seconds"),
                "median_completed_sampling_seconds": _median(
                    results, "completed_sampling_seconds"
                ),
                "median_dispatch_seconds": _median(results, "dispatch_seconds"),
                "median_materialize_seconds": _median(results, "materialize_seconds"),
                "median_min_ess_bulk": _median(results, "min_ess_bulk"),
                "median_max_rhat": _median(results, "max_rhat"),
                "total_divergences": int(
                    sum(int(r.get("n_divergences", 0)) for r in successes)
                ),
                "sequence_completed_sampling_medians": _sequence_medians(results),
                "fold_records": fold_records,
                "pymc_versions": _runtime_values(results, "pymc_version"),
                "jax_versions": _runtime_values(results, "jax_version"),
                "blackjax_versions": _runtime_values(results, "blackjax_version"),
                "sampling_configurations": _runtime_values(
                    results, "sampling_configuration"
                ),
                "jax_device_probe_timing": "post_sampling_only",
                "rows_per_individual": args.rows_per_individual,
                "predictors": args.predictors,
                "bin_width": args.bin_width,
                "random_slopes": args.random_slopes,
                "draws": args.draws,
                "tune": args.tune,
                "chains": args.chains,
                "cores_argument": args.cores,
                "failure_examples": failure_examples,
            }

            append_benchmark_record(
                make_benchmark_record(
                    "bayesian_rsf_chain_method",
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
                f"sampler={args.sampler} chain_method={chain_method} "
                f"workers={workers} repeat={repeat} status={status} "
                f"completed={completed}/{len(ids)} wall={wall:.3f}s "
                f"fold={_format_seconds(metadata['median_fold_seconds'])} "
                f"sequence={metadata['sequence_completed_sampling_medians']}"
            )
            print(
                f"  PyMC={metadata['pymc_versions']} JAX={metadata['jax_versions']} "
                f"BlackJAX={metadata['blackjax_versions']} "
                f"configs={metadata['sampling_configurations']}"
            )
            if successes:
                device_counts = sorted(
                    {
                        int(r["jax_local_device_count"])
                        for r in successes
                        if r.get("jax_local_device_count") is not None
                    }
                )
                print(f"  post-sampling JAX local device counts={device_counts}")
            if failure_examples:
                print("  failure examples:")
                for example in failure_examples:
                    print(f"    - {example}")
                first_traceback = next(
                    (r.get("traceback") for r in failures if r.get("traceback")), None
                )
                if first_traceback:
                    print("  first traceback:")
                    print(first_traceback.rstrip())
    finally:
        client.close()
        cluster.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose Bayesian RSF JAX chain execution without preinitializing JAX."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=parse_positive_ints, default=[4, 12])
    parser.add_argument("--chain-methods", default="default,parallel,vectorized")
    parser.add_argument("--folds", type=int, default=24)
    parser.add_argument("--rows-per-individual", type=int, default=500)
    parser.add_argument("--predictors", type=int, default=3)
    parser.add_argument("--sampler", choices=("blackjax", "numpyro"), default="blackjax")
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--random-slopes", type=int, default=1)
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--tune", type=int, default=250)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    args = parser.parse_args()

    chain_methods = [item.strip() for item in args.chain_methods.split(",") if item.strip()]
    unknown = sorted(set(chain_methods).difference({"default", "parallel", "vectorized"}))
    if unknown:
        parser.error(f"Unknown --chain-methods values: {unknown}")
    if args.random_slopes < 0 or args.random_slopes > args.predictors:
        parser.error("--random-slopes must be between 0 and --predictors")
    if args.cores <= 0 or args.chains <= 0 or args.repeats <= 0:
        parser.error("cores, chains and repeats must be positive")

    args.output = args.output.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cache = args.work_dir / (
        f"cv-cache-{args.folds}x{args.rows_per_individual}-{args.predictors}p-seed{args.seed}"
    )
    prepared = cvbench._prepare_synthetic_cache(
        cache,
        folds=args.folds,
        rows_per_individual=args.rows_per_individual,
        predictors=args.predictors,
        seed=args.seed,
        overwrite=args.overwrite_cache,
    )

    for chain_method in chain_methods:
        for workers in args.workers:
            _run_point(
                args,
                workers=workers,
                chain_method=chain_method,
                prepared=prepared,
                output=args.output,
            )


if __name__ == "__main__":
    main()
