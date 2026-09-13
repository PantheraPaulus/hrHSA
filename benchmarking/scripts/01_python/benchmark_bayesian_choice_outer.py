"""Benchmark outer-model throughput for Bayesian SSF/iSSF on CPU clusters.

Independent synthetic hierarchical choice models are the distributed unit. Each
fit retains the requested number of statistical chains while ``cores_per_fit``
controls how many chains a CPU-native sampler may execute concurrently. This is
an execution-geometry benchmark; it does not stand in for ecological CV.

Distributed callables intentionally depend only on installed package modules,
not sibling benchmark scripts, so Dask scheduler/worker ``sys.path`` differences
cannot break task deserialization on HPC systems.

Completed model records are streamed to a sidecar JSONL as they arrive. Short
Slurm diagnostics can therefore retain useful per-model evidence even if the
allocation ends before the complete fixed-size ensemble has finished.
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
from hsa.rsf.bayesian_hpc import _materialize_inference_tree
from hsa.ssf.bayesian import build_hierarchical_ssf_model
from hsa.ssf.data import build_ssf_choice_arrays


def _distribution_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _fit_choice_model(
    *,
    model_index: int,
    analysis: str,
    n_strata: int,
    n_choices: int,
    predictors: int,
    individuals: int,
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
        import arviz as az
        import pandas as pd
        import pymc as pm
        from dask.distributed import get_worker
        from scipy.special import logsumexp

        worker = get_worker()
        worker_address = worker.address
        worker_name = str(worker.name)

        prepare_started = perf_counter()
        rng = np.random.default_rng(data_seed)
        if analysis == "ssf":
            names = [f"x{i}" for i in range(predictors)]
        else:
            if predictors < 3:
                raise ValueError("iSSF benchmark requires at least three predictors")
            names = [*[f"x{i}" for i in range(predictors - 2)], "log_sl", "cos_ta"]

        X = rng.normal(size=(n_strata, n_choices, predictors)).astype(np.float32)
        if analysis == "issf":
            X[:, :, names.index("log_sl")] = rng.normal(
                0.0, 0.7, size=(n_strata, n_choices)
            )
            X[:, :, names.index("cos_ta")] = rng.uniform(
                -1.0, 1.0, size=(n_strata, n_choices)
            )
            offset = rng.normal(
                0.0, 0.35, size=(n_strata, n_choices)
            ).astype(np.float32)
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

        frame_data: dict[str, Any] = {
            "id": np.repeat(id_by_stratum, n_choices),
            "stratum_id": np.repeat(strata, n_choices),
            "candidate_id": np.tile(
                np.arange(n_choices, dtype=np.int16), n_strata
            ),
            "used": used.reshape(-1),
        }
        flat = X.reshape(-1, predictors)
        for index, name in enumerate(names):
            frame_data[name] = flat[:, index]
        if offset_col is not None:
            frame_data[offset_col] = offset.reshape(-1)

        frame = pd.DataFrame(frame_data)
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
        prepare_seconds = perf_counter() - prepare_started

        stage = "build_model"
        model_started = perf_counter()
        model = build_hierarchical_ssf_model(arrays)
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
                max_tree_depth = float(np.asarray(sample_stats["tree_depth"]).max())

        diagnostics = {
            "min_ess_bulk": float(np.nanmin(ess_bulk)),
            "median_ess_bulk": float(np.nanmedian(ess_bulk)),
            "min_ess_tail": float(np.nanmin(ess_tail)),
            "median_ess_tail": float(np.nanmedian(ess_tail)),
            "min_ess_bulk_per_completed_second": float(np.nanmin(ess_bulk))
            / completed_sampling_seconds,
            "median_ess_bulk_per_completed_second": float(np.nanmedian(ess_bulk))
            / completed_sampling_seconds,
            "min_ess_tail_per_completed_second": float(np.nanmin(ess_tail))
            / completed_sampling_seconds,
            "median_ess_tail_per_completed_second": float(np.nanmedian(ess_tail))
            / completed_sampling_seconds,
            "max_rhat": float(np.nanmax(rhat)),
            "n_divergences": divergences,
            "mean_n_steps": mean_n_steps,
            "max_tree_depth": max_tree_depth,
        }
        diagnostics_seconds = perf_counter() - diagnostics_started

        return {
            "model_index": int(model_index),
            "status": "success",
            "wall_seconds": perf_counter() - started,
            "prepare_seconds": prepare_seconds,
            "model_seconds": model_seconds,
            "dispatch_seconds": dispatch_seconds,
            "materialize_seconds": materialize_seconds,
            "completed_sampling_seconds": completed_sampling_seconds,
            "diagnostics_seconds": diagnostics_seconds,
            "worker_address": worker_address,
            "worker_name": worker_name,
            "worker_pid": os.getpid(),
            "pymc_version": _distribution_version("pymc"),
            "nutpie_version": _distribution_version("nutpie"),
            **diagnostics,
            "error": None,
        }
    except Exception as exc:
        return {
            "model_index": int(model_index),
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
        description="Benchmark outer-parallel hierarchical Bayesian SSF/iSSF fits."
    )
    parser.add_argument("--analysis", choices=("ssf", "issf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--cores-per-fit", type=int, required=True)
    parser.add_argument("--models", type=int, required=True)
    parser.add_argument("--strata", type=int, default=10_000)
    parser.add_argument("--choices", type=int, default=11)
    parser.add_argument("--predictors", type=int, default=None)
    parser.add_argument("--individuals", type=int, default=20)
    parser.add_argument("--sampler", default="nutpie")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=10_000)
    parser.add_argument("--memory-monitor-interval", type=float, default=5.0)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.predictors is None:
        args.predictors = 6 if args.analysis == "ssf" else 14

    for name in (
        "workers",
        "cores_per_fit",
        "models",
        "strata",
        "choices",
        "predictors",
        "individuals",
        "draws",
        "tune",
        "chains",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.analysis == "issf" and args.predictors < 3:
        parser.error("iSSF requires at least three predictors")
    if args.models < args.workers:
        parser.error("--models must be >= --workers for an outer-throughput point")
    if args.memory_monitor_interval <= 0:
        parser.error("--memory-monitor-interval must be positive")

    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_output = args.output.with_name(f"{args.output.stem}.tasks.jsonl")
    progress_output.unlink(missing_ok=True)

    from dask.distributed import Client, LocalCluster, as_completed

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
            raise RuntimeError(
                f"Expected exactly {args.workers} workers; observed {observed}."
            )

        calls = [
            {
                "model_index": index,
                "analysis": args.analysis,
                "n_strata": args.strata,
                "n_choices": args.choices,
                "predictors": args.predictors,
                "individuals": args.individuals,
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

        results: list[dict[str, Any]] = []
        progress_started = perf_counter()
        with benchmark_timer(
            client=client,
            memory_interval_seconds=args.memory_monitor_interval,
        ) as timer:
            futures = [client.submit(_fit_choice_model, pure=False, **call) for call in calls]
            for completion_index, (_, result) in enumerate(
                as_completed(futures, with_results=True, raise_errors=True),
                start=1,
            ):
                results.append(result)
                append_benchmark_record(
                    {
                        "completion_index": completion_index,
                        "elapsed_seconds": perf_counter() - progress_started,
                        "analysis": args.analysis,
                        "workers": args.workers,
                        "cores_per_fit": args.cores_per_fit,
                        "models_requested": args.models,
                        "strata": args.strata,
                        "choices": args.choices,
                        "predictors": args.predictors,
                        "draws": args.draws,
                        "tune": args.tune,
                        "chains": args.chains,
                        **result,
                    },
                    progress_output,
                )
                if completion_index % 10 == 0 or completion_index == args.models:
                    print(
                        f"progress: {completion_index}/{args.models} models completed "
                        f"in {perf_counter() - progress_started:.1f}s",
                        flush=True,
                    )

        successes = [r for r in results if r.get("status") == "success"]
        failures = [r for r in results if r.get("status") != "success"]
        completed = len(successes)
        wall = float(timer["wall_seconds"])
        metadata = {
            "campaign": "bayesian_choice_outer_v1",
            "analysis": args.analysis,
            "sampler": args.sampler,
            "models_requested": args.models,
            "models_completed": completed,
            "models_failed": len(failures),
            "models_per_second": completed / wall if wall > 0 else np.nan,
            "workers": args.workers,
            "cores_per_fit": args.cores_per_fit,
            "nominal_active_cores": args.workers * args.cores_per_fit,
            "chains": args.chains,
            "draws": args.draws,
            "tune": args.tune,
            "target_accept": args.target_accept,
            "strata": args.strata,
            "choices": args.choices,
            "predictors": args.predictors,
            "individuals": args.individuals,
            "memory_monitor_interval_seconds": args.memory_monitor_interval,
            "progress_output": str(progress_output),
            "median_fit_wall_seconds": _median(results, "wall_seconds"),
            "median_completed_sampling_seconds": _median(
                results, "completed_sampling_seconds"
            ),
            "median_min_ess_bulk": _median(results, "min_ess_bulk"),
            "median_median_ess_bulk": _median(results, "median_ess_bulk"),
            "median_min_ess_bulk_per_completed_second": _median(
                results, "min_ess_bulk_per_completed_second"
            ),
            "median_max_rhat": _median(results, "max_rhat"),
            "total_divergences": int(
                sum(int(r.get("n_divergences", 0)) for r in successes)
            ),
            "pymc_versions": sorted(
                {str(r["pymc_version"]) for r in successes if r.get("pymc_version")}
            ),
            "nutpie_versions": sorted(
                {str(r["nutpie_version"]) for r in successes if r.get("nutpie_version")}
            ),
            "failure_examples": [r.get("error") for r in failures[:3]],
            "observed_workers": observed,
        }

        append_benchmark_record(
            make_benchmark_record(
                f"bayesian_{args.analysis}_outer",
                wall,
                rows=args.models,
                workers=args.workers,
                threads_per_worker=1,
                bytes_processed=None,
                metadata=metadata,
                operation_memory=timer,
                client=client,
            ),
            args.output,
        )

        status = "ok" if not failures else ("failed" if not successes else "partial")
        print(
            f"{args.analysis}: workers={args.workers} cores/fit={args.cores_per_fit} "
            f"models={completed}/{args.models} status={status} wall={wall:.3f}s "
            f"median_fit={metadata['median_completed_sampling_seconds']} "
            f"minESS={metadata['median_min_ess_bulk']} "
            f"Rhat={metadata['median_max_rhat']} div={metadata['total_divergences']}"
        )
        print(f"per-model progress: {progress_output}")
        if failures:
            print("  failures:", metadata["failure_examples"])
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
