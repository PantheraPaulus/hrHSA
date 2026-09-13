"""Benchmark explicit outer scheduling of independent BlackJAX chains.

This prototype flattens Bayesian RSF inference from

    outer fold task -> BlackJAX chains inside JAX pmap

to

    outer Dask task -> one BlackJAX chain

and recombines the independent one-chain InferenceData objects per fold before
computing multi-chain diagnostics.  The benchmark intentionally includes repeated
model construction, JAX compilation, Dask result transfer, recombination and
diagnostics so it measures the real cost of making chain concurrency visible to
the outer scheduler.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any

import numpy as np

import benchmark_cv_scaling as cvbench
from hsa.compute import append_benchmark_record, benchmark_timer, make_benchmark_record
from hsa.rsf.bayesian import build_bayesian_rsf_model, prepare_bayesian_rsf_data
from hsa.rsf.bayesian_cv import _sampling_diagnostics
from hsa.rsf.bayesian_hpc import _materialize_inference_tree
from hsa.rsf.cv_parallel import execute_fold_calls


_WORKER_CHAIN_SEQUENCE = 0


def _single_chain_task(
    *,
    prepared_root: str,
    fold_index: int,
    heldout_id: Any,
    chain_index: int,
    predictors: list[str],
    bin_width: float,
    random_slopes: int,
    sampler: str,
    draws: int,
    tune: int,
    target_accept: float,
    seed: int,
) -> dict[str, Any]:
    """Fit exactly one chain for one held-out fold and materialize it on-worker."""
    global _WORKER_CHAIN_SEQUENCE
    _WORKER_CHAIN_SEQUENCE += 1
    sequence = _WORKER_CHAIN_SEQUENCE
    started = perf_counter()
    stage = "worker_setup"
    worker_address = None
    worker_name = None

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

        sampling: dict[str, Any] = {
            "draws": draws,
            "tune": tune,
            "chains": 1,
            "cores": 1,
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

        model_meta = bayes_data["meta"]["_model"]
        return {
            "fold_index": int(fold_index),
            "heldout_ID": heldout_id,
            "chain_index": int(chain_index),
            "status": "success",
            "stage": "complete",
            "wall_seconds": perf_counter() - started,
            "load_seconds": load_seconds,
            "prepare_seconds": prepare_seconds,
            "model_seconds": model_seconds,
            "dispatch_seconds": dispatch_seconds,
            "materialize_seconds": materialize_seconds,
            "completed_sampling_seconds": completed_sampling_seconds,
            "n_raw": int(model_meta["n_raw"]),
            "n_aggregated": int(model_meta["n_aggregated"]),
            "worker_address": worker_address,
            "worker_name": worker_name,
            "worker_pid": os.getpid(),
            "worker_chain_sequence": sequence,
            "idata": idata,
            "error": None,
            "traceback": None,
        }
    except Exception as exc:
        return {
            "fold_index": int(fold_index),
            "heldout_ID": heldout_id,
            "chain_index": int(chain_index),
            "status": "failed",
            "stage": stage,
            "wall_seconds": perf_counter() - started,
            "worker_address": worker_address,
            "worker_name": worker_name,
            "worker_pid": os.getpid(),
            "worker_chain_sequence": sequence,
            "idata": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _combine_chain_idatas(idatas: list[Any]):
    """Combine one-chain InferenceData objects into one multi-chain object."""
    if not idatas:
        raise ValueError("At least one InferenceData object is required.")

    try:
        import arviz as az
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - benchmark environment
        raise ImportError("ArviZ and xarray are required for chain recombination.") from exc

    reference_groups = list(idatas[0].groups())
    for idata in idatas[1:]:
        if list(idata.groups()) != reference_groups:
            raise ValueError("One-chain InferenceData objects expose different groups.")

    combined_groups: dict[str, Any] = {}
    for group in reference_groups:
        datasets = [getattr(idata, group) for idata in idatas]
        chain_flags = ["chain" in dataset.dims for dataset in datasets]
        if all(chain_flags):
            normalized = []
            for chain_index, dataset in enumerate(datasets):
                if int(dataset.sizes["chain"]) != 1:
                    raise ValueError(
                        f"Expected one chain in group {group!r}; "
                        f"got {dataset.sizes['chain']}."
                    )
                normalized.append(dataset.assign_coords(chain=[chain_index]))
            combined_groups[group] = xr.concat(
                normalized,
                dim="chain",
                combine_attrs="override",
            )
        elif any(chain_flags):
            raise ValueError(f"Inconsistent chain dimension in group {group!r}.")
        else:
            # observed_data / constant_data are identical for all chains.
            combined_groups[group] = datasets[0].copy(deep=False)

    return az.InferenceData(**combined_groups)


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
            int(result["worker_chain_sequence"])
            for result in results
            if result.get("status") == "success"
        }
    )
    for sequence in sequences:
        values = [
            float(result["completed_sampling_seconds"])
            for result in results
            if result.get("status") == "success"
            and int(result["worker_chain_sequence"]) == sequence
        ]
        if values:
            output[f"sequence_{sequence}"] = float(np.median(values))
    return output


def _run_campaign(args, client, prepared, output: Path) -> None:
    ids = prepared.individuals[: args.folds]
    names = list(prepared.predictors)

    for repeat in range(1, args.repeats + 1):
        calls = [
            {
                "prepared_root": str(prepared.root),
                "fold_index": fold_index,
                "heldout_id": heldout_id,
                "chain_index": chain_index,
                "predictors": names,
                "bin_width": args.bin_width,
                "random_slopes": args.random_slopes,
                "sampler": args.sampler,
                "draws": args.draws,
                "tune": args.tune,
                "target_accept": args.target_accept,
                "seed": (
                    args.seed
                    + repeat
                    + 100_000 * fold_index
                    + 1_000 * chain_index
                ),
            }
            for fold_index, heldout_id in enumerate(ids)
            for chain_index in range(args.chains_per_fold)
        ]

        with benchmark_timer(client=client) as timer:
            chain_results = execute_fold_calls(_single_chain_task, calls, client=client)

            combine_started = perf_counter()
            fold_results: list[dict[str, Any]] = []
            for fold_index, heldout_id in enumerate(ids):
                records = [
                    result
                    for result in chain_results
                    if int(result["fold_index"]) == fold_index
                ]
                successes = [r for r in records if r.get("status") == "success"]
                failures = [r for r in records if r.get("status") != "success"]
                fold_started = perf_counter()

                if failures or len(successes) != args.chains_per_fold:
                    fold_results.append(
                        {
                            "fold_index": fold_index,
                            "heldout_ID": heldout_id,
                            "status": "failed",
                            "chains_completed": len(successes),
                            "error": (
                                failures[0].get("error")
                                if failures
                                else "missing one-chain result"
                            ),
                        }
                    )
                    continue

                try:
                    idatas = [
                        result["idata"]
                        for result in sorted(
                            successes,
                            key=lambda result: int(result["chain_index"]),
                        )
                    ]
                    idata = _combine_chain_idatas(idatas)
                    diagnostics = _sampling_diagnostics(idata)
                    fold_results.append(
                        {
                            "fold_index": fold_index,
                            "heldout_ID": heldout_id,
                            "status": "success",
                            "chains_completed": len(idatas),
                            "combine_diagnostics_seconds": perf_counter() - fold_started,
                            **diagnostics,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    fold_results.append(
                        {
                            "fold_index": fold_index,
                            "heldout_ID": heldout_id,
                            "status": "failed",
                            "chains_completed": len(successes),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

            combine_seconds = perf_counter() - combine_started

            # Drop posterior payloads before record assembly.
            for result in chain_results:
                result.pop("idata", None)

        wall = float(timer["wall_seconds"])
        chain_successes = [r for r in chain_results if r.get("status") == "success"]
        chain_failures = [r for r in chain_results if r.get("status") != "success"]
        fold_successes = [r for r in fold_results if r.get("status") == "success"]
        fold_failures = [r for r in fold_results if r.get("status") != "success"]

        metadata = {
            "campaign": "bayesian-rsf-explicit-chain-scheduling-v1",
            "scheduler_geometry": "one_dask_task_per_chain",
            "sampler": args.sampler,
            "workers": args.workers,
            "folds_requested": len(ids),
            "folds_completed": len(fold_successes),
            "folds_failed": len(fold_failures),
            "chains_per_fold": args.chains_per_fold,
            "chain_tasks_requested": len(calls),
            "chain_tasks_completed": len(chain_successes),
            "chain_tasks_failed": len(chain_failures),
            "folds_per_second": len(fold_successes) / wall if wall > 0 else np.nan,
            "chains_per_second": len(chain_successes) / wall if wall > 0 else np.nan,
            "median_chain_wall_seconds": _median(chain_results, "wall_seconds"),
            "median_chain_dispatch_seconds": _median(chain_results, "dispatch_seconds"),
            "median_chain_materialize_seconds": _median(
                chain_results, "materialize_seconds"
            ),
            "median_chain_completed_sampling_seconds": _median(
                chain_results, "completed_sampling_seconds"
            ),
            "worker_sequence_completed_sampling_medians": _sequence_medians(
                chain_results
            ),
            "combine_and_diagnostics_seconds": combine_seconds,
            "median_fold_combine_diagnostics_seconds": _median(
                fold_results, "combine_diagnostics_seconds"
            ),
            "median_min_ess_bulk": _median(fold_results, "min_ess_bulk"),
            "median_max_rhat": _median(fold_results, "max_rhat"),
            "total_divergences": int(
                sum(int(result.get("n_divergences", 0)) for result in fold_successes)
            ),
            "rows_per_individual": args.rows_per_individual,
            "predictors": args.predictors,
            "bin_width": args.bin_width,
            "random_slopes": args.random_slopes,
            "draws": args.draws,
            "tune": args.tune,
            "target_accept": args.target_accept,
            "failure_examples": [
                result.get("error")
                for result in (chain_failures + fold_failures)[:3]
            ],
        }

        append_benchmark_record(
            make_benchmark_record(
                "bayesian_rsf_explicit_chain_scheduling",
                wall,
                rows=len(ids),
                workers=args.workers,
                threads_per_worker=1,
                bytes_processed=int(prepared.manifest["n_rows"].sum()),
                metadata=metadata,
                operation_memory=timer,
                client=client,
            ),
            output,
        )

        chain_median = metadata["median_chain_completed_sampling_seconds"]
        chain_text = "NA" if chain_median is None else f"{chain_median:.3f}s"
        print(
            f"explicit-chains: workers={args.workers} "
            f"folds={len(fold_successes)}/{len(ids)} "
            f"chains={len(chain_successes)}/{len(calls)} "
            f"wall={wall:.3f}s folds/s={metadata['folds_per_second']:.4f} "
            f"chain={chain_text} combine={combine_seconds:.3f}s"
        )
        if metadata["failure_examples"]:
            print("  failures:", metadata["failure_examples"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one-Dask-task-per-BlackJAX-chain scheduling for Bayesian RSF."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--folds", type=int, default=24)
    parser.add_argument("--chains-per-fold", type=int, default=4)
    parser.add_argument("--rows-per-individual", type=int, default=500)
    parser.add_argument("--predictors", type=int, default=3)
    parser.add_argument("--sampler", default="blackjax")
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--random-slopes", type=int, default=1)
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--tune", type=int, default=250)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    args = parser.parse_args()

    for name in (
        "workers",
        "folds",
        "chains_per_fold",
        "rows_per_individual",
        "predictors",
        "draws",
        "tune",
        "repeats",
    ):
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

    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    )
    client = Client(cluster)
    try:
        client.wait_for_workers(args.workers, timeout=args.worker_startup_timeout)
        observed = len(client.scheduler_info().get("workers", {}))
        if observed != args.workers:
            raise RuntimeError(
                f"Expected exactly {args.workers} workers; observed {observed}."
            )
        _run_campaign(args, client, prepared, args.output)
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
