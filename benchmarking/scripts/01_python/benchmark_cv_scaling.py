"""Benchmark outer-fold cross-validation scaling for hrHSA inference.

Independent held-out folds are the distributed unit, while native BLAS/PyMC work
inside a fold is explicitly bounded. The synthetic per-individual Parquet cache
mimics a PreparedDataset so shared-filesystem reads, design construction and model
fitting are represented without depending on a specific ecological dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from hsa import FeatureSpec
from hsa.compute import append_benchmark_record, benchmark_timer, make_benchmark_record
from hsa.compute.prepared import PreparedDataset
from hsa.rsf.bayesian import build_bayesian_rsf_model, prepare_bayesian_rsf_data
from hsa.rsf.bayesian_cv import _sampling_diagnostics
from hsa.rsf.bayesian_hpc import _materialize_inference_tree
from hsa.rsf.cv_parallel import execute_fold_calls
from hsa.rsf.model import fit_rsf


def _frame_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(index=True, deep=True).sum())


def _spec(names: list[str]) -> FeatureSpec:
    return FeatureSpec(
        linear=names,
        quadratic=names[:1],
        interactions=[(names[0], names[1])] if len(names) > 1 else [],
        add_const=True,
    )


def _cache_metadata(*, folds: int, rows_per_individual: int, predictors: int, seed: int) -> dict[str, Any]:
    names = [f"x{i}" for i in range(predictors)]
    return {
        "format": "hrHSA-cv-scaling-v1",
        "id_col": "individual-local-identifier",
        "predictors": names,
        "sampling_factor": 1,
        "thin_dt": None,
        "folds": folds,
        "rows_per_individual": rows_per_individual,
        "seed": seed,
    }


def _cache_matches(root: Path, expected: dict[str, Any]) -> bool:
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    keys = ("format", "id_col", "predictors", "folds", "rows_per_individual", "seed")
    return all(observed.get(key) == expected.get(key) for key in keys)


def _prepare_synthetic_cache(
    root: Path,
    *,
    folds: int,
    rows_per_individual: int,
    predictors: int,
    seed: int,
    overwrite: bool = False,
) -> PreparedDataset:
    expected = _cache_metadata(
        folds=folds,
        rows_per_individual=rows_per_individual,
        predictors=predictors,
        seed=seed,
    )
    if root.exists() and not overwrite and _cache_matches(root, expected):
        return PreparedDataset.open(root)

    if root.exists():
        shutil.rmtree(root)
    partitions = root / "individuals"
    partitions.mkdir(parents=True, exist_ok=True)

    names = list(expected["predictors"])
    beta = np.linspace(0.55, -0.25, predictors, dtype=np.float64)
    manifest_rows: list[dict[str, Any]] = []

    for index in range(folds):
        individual_id = f"id-{index:04d}"
        rng = np.random.default_rng(seed + 10_000 * index)
        X = rng.normal(size=(rows_per_individual, predictors)).astype(np.float64, copy=False)
        individual_effect = float(rng.normal(0.0, 0.35))
        eta = -2.0 + X @ beta + individual_effect
        used = rng.binomial(1, expit(eta)).astype(np.int8)
        if rows_per_individual >= 2:
            used[0] = 1
            used[1] = 0

        data: dict[str, Any] = {name: X[:, column] for column, name in enumerate(names)}
        data["used"] = used
        data["individual-local-identifier"] = individual_id
        frame = pd.DataFrame(data)

        filename = f"{index:05d}-{individual_id}.parquet"
        relative = Path("individuals") / filename
        frame.to_parquet(root / relative, index=False)
        manifest_rows.append(
            {
                "individual_id": individual_id,
                "file": relative.as_posix(),
                "n_rows": int(len(frame)),
                "n_used": int(used.sum()),
                "n_available": int((1 - used).sum()),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(root / "manifest.parquet", index=False)
    manifest.to_csv(root / "manifest.csv", index=False)
    metadata = {
        **expected,
        "n_individuals": int(len(manifest)),
        "n_rows": int(manifest["n_rows"].sum()),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return PreparedDataset.open(root)


def _frequentist_fold(
    *,
    prepared_root: str,
    heldout_id: Any,
    spec: FeatureSpec,
    method: str,
    maxiter: int,
    blas_threads: int,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        prepared = PreparedDataset.open(prepared_root)
        train_df = prepared.load(exclude=heldout_id)
        load_seconds = perf_counter() - started
        fit_started = perf_counter()
        with threadpool_limits(limits=blas_threads):
            model, _, _, _ = fit_rsf(
                train_df,
                spec,
                method=method,
                fit_kwargs={"maxiter": maxiter},
            )
        fit_seconds = perf_counter() - fit_started
        retvals = getattr(model, "mle_retvals", None) or {}
        return {
            "heldout_ID": heldout_id,
            "status": "success",
            "wall_seconds": perf_counter() - started,
            "load_seconds": load_seconds,
            "fit_seconds": fit_seconds,
            "n_train": int(len(train_df)),
            "converged": bool(retvals.get("converged", True)),
            "llf": float(model.llf),
            "error": None,
        }
    except Exception as exc:
        return {
            "heldout_ID": heldout_id,
            "status": "failed",
            "wall_seconds": perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _bayesian_fold(
    *,
    prepared_root: str,
    heldout_id: Any,
    predictors: list[str],
    bin_width: float,
    random_slopes: int,
    sampler: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    seed: int,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        import inspect
        import pymc as pm

        prepared = PreparedDataset.open(prepared_root)
        train_df = prepared.load(exclude=heldout_id)
        load_seconds = perf_counter() - started

        prepare_started = perf_counter()
        bayes_data = prepare_bayesian_rsf_data(
            train_df,
            predictors,
            id_col="individual-local-identifier",
            binning={name: bin_width for name in predictors},
        )
        prepare_seconds = perf_counter() - prepare_started

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
            "cores": cores,
            "target_accept": target_accept,
            "progressbar": False,
            "return_inferencedata": True,
            "random_seed": seed,
        }
        sample_signature = inspect.signature(pm.sample)
        if "compute_convergence_checks" in sample_signature.parameters:
            sampling["compute_convergence_checks"] = False
        if "blas_cores" in sample_signature.parameters:
            sampling["blas_cores"] = 1
        if "nuts_sampler" in sample_signature.parameters:
            sampling["nuts_sampler"] = sampler
        elif sampler != "pymc":
            raise RuntimeError("this PyMC version has no nuts_sampler argument")

        sample_started = perf_counter()
        with model:
            raw_idata = pm.sample(**sampling)
        dispatch_seconds = perf_counter() - sample_started

        materialize_started = perf_counter()
        idata = _materialize_inference_tree(raw_idata)
        materialize_seconds = perf_counter() - materialize_started
        completed_sampling_seconds = dispatch_seconds + materialize_seconds

        diagnostics_started = perf_counter()
        diagnostics = _sampling_diagnostics(idata)
        diagnostics_seconds = perf_counter() - diagnostics_started
        model_meta = bayes_data["meta"]["_model"]

        return {
            "heldout_ID": heldout_id,
            "status": "success",
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
            **diagnostics,
            "error": None,
        }
    except Exception as exc:
        return {
            "heldout_ID": heldout_id,
            "status": "failed",
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


def _run_campaign(args, client, prepared: PreparedDataset, output: Path) -> None:
    ids = prepared.individuals[: args.folds]
    names = list(prepared.predictors)
    spec = _spec(names)

    for repeat in range(1, args.repeats + 1):
        if args.mode == "frequentist":
            calls = [
                {
                    "prepared_root": str(prepared.root),
                    "heldout_id": heldout_id,
                    "spec": spec,
                    "method": args.method,
                    "maxiter": args.maxiter,
                    "blas_threads": args.blas_threads,
                }
                for heldout_id in ids
            ]
            fold_fn = _frequentist_fold
        else:
            calls = [
                {
                    "prepared_root": str(prepared.root),
                    "heldout_id": heldout_id,
                    "predictors": names,
                    "bin_width": args.bin_width,
                    "random_slopes": args.random_slopes,
                    "sampler": args.sampler,
                    "draws": args.draws,
                    "tune": args.tune,
                    "chains": args.chains,
                    "cores": args.cores,
                    "target_accept": args.target_accept,
                    "seed": args.seed + repeat + 100_000 * fold_id,
                }
                for fold_id, heldout_id in enumerate(ids)
            ]
            fold_fn = _bayesian_fold

        with benchmark_timer(client=client) as campaign_timer:
            results = execute_fold_calls(fold_fn, calls, client=client)
        wall_seconds = campaign_timer["wall_seconds"]

        successes = [result for result in results if result.get("status") == "success"]
        failures = [result for result in results if result.get("status") != "success"]
        completed = len(successes)
        folds_per_second = completed / wall_seconds if wall_seconds > 0 else np.nan
        allocated_core_seconds_per_fold = (
            wall_seconds * args.workers / completed if completed else np.nan
        )

        metadata: dict[str, Any] = {
            "campaign": "cv_outer_scaling",
            "mode": args.mode,
            "variant": (
                f"{args.method}_{args.workers}w"
                if args.mode == "frequentist"
                else f"{args.sampler}_{args.workers}w"
            ),
            "repeat": repeat,
            "folds_requested": len(ids),
            "folds_completed": completed,
            "folds_failed": len(failures),
            "folds_per_second": folds_per_second,
            "allocated_core_seconds_per_completed_fold": allocated_core_seconds_per_fold,
            "workers": args.workers,
            "threads_per_worker": 1,
            "rows_per_individual": args.rows_per_individual,
            "predictors": args.predictors,
            "median_fold_seconds": _median(results, "wall_seconds"),
            "median_load_seconds": _median(results, "load_seconds"),
            "memory_monitor_interval_seconds": 0.25,
            "failure_examples": [result.get("error") for result in failures[:3]],
        }
        if args.mode == "frequentist":
            metadata.update(
                optimizer=args.method,
                blas_threads=args.blas_threads,
                maxiter=args.maxiter,
                median_fit_seconds=_median(results, "fit_seconds"),
            )
        else:
            metadata.update(
                nuts_sampler=args.sampler,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores_per_fold=args.cores,
                bin_width=args.bin_width,
                random_slopes=args.random_slopes,
                median_prepare_seconds=_median(results, "prepare_seconds"),
                median_model_seconds=_median(results, "model_seconds"),
                median_dispatch_seconds=_median(results, "dispatch_seconds"),
                median_materialize_seconds=_median(results, "materialize_seconds"),
                median_completed_sampling_seconds=_median(
                    results, "completed_sampling_seconds"
                ),
                median_diagnostics_seconds=_median(results, "diagnostics_seconds"),
                median_max_rhat=_median(results, "max_rhat"),
                median_min_ess_bulk=_median(results, "min_ess_bulk"),
                total_divergences=int(
                    sum(int(result.get("n_divergences", 0)) for result in successes)
                ),
            )

        record = make_benchmark_record(
            f"cv_{args.mode}_outer_scaling",
            wall_seconds,
            rows=len(ids),
            workers=args.workers,
            threads_per_worker=1,
            bytes_processed=int(prepared.manifest["n_rows"].sum()),
            metadata=metadata,
            operation_memory=campaign_timer,
            client=client,
        )
        append_benchmark_record(record, output)
        print(
            f"{args.mode}: workers={args.workers}, repeat={repeat}, "
            f"completed={completed}/{len(ids)}, wall={wall_seconds:.3f}s, "
            f"folds/s={folds_per_second:.4f}"
        )
        if failures:
            print("  failures:", metadata["failure_examples"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark outer-fold CV scaling on a local or existing Dask cluster."
    )
    parser.add_argument("mode", choices=("frequentist", "bayesian"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--scheduler-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--folds", type=int, default=28)
    parser.add_argument("--rows-per-individual", type=int, default=2_000)
    parser.add_argument("--predictors", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)

    parser.add_argument("--method", default="lbfgs")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--blas-threads", type=int, default=1)

    parser.add_argument("--sampler", default="pymc")
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--random-slopes", type=int, default=1)
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--tune", type=int, default=250)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=0.9)
    args = parser.parse_args()

    for name in ("workers", "folds", "rows_per_individual", "predictors", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.random_slopes < 0 or args.random_slopes > args.predictors:
        parser.error("--random-slopes must be between 0 and --predictors")
    if args.blas_threads <= 0 or args.cores <= 0 or args.chains <= 0:
        parser.error("BLAS threads, Bayesian cores and chains must be positive")

    args.output = args.output.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cache = args.work_dir / (
        f"cv-cache-{args.folds}x{args.rows_per_individual}-{args.predictors}p-seed{args.seed}"
    )
    prepared = _prepare_synthetic_cache(
        cache,
        folds=args.folds,
        rows_per_individual=args.rows_per_individual,
        predictors=args.predictors,
        seed=args.seed,
        overwrite=args.overwrite_cache,
    )

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
            raise RuntimeError(
                f"Expected exactly {args.workers} workers for a scaling point; observed {observed}."
            )
        _run_campaign(args, client, prepared, args.output)
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
