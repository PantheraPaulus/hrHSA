"""Diagnose BlackJAX/PyMC outer-fold CPU contention on one machine.

This benchmark deliberately reuses the synthetic PreparedDataset and Bayesian fold
implementation from ``benchmark_cv_scaling.py``. The only experimental factor is
whether each Dask worker process is left with its inherited CPU affinity or pinned
to one distinct physical CPU before PyMC/JAX initializes.

The goal is mechanistic: if BlackJAX fold time remains near its one-worker value
when workers are pinned, the historical outer-scaling loss was CPU-pool contention
rather than a statistical-model bottleneck.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

import benchmark_cv_scaling as cvbench
from hsa.compute import append_benchmark_record, benchmark_timer, make_benchmark_record
from hsa.compute.workloads import parse_positive_ints
from hsa.rsf.cv_parallel import execute_fold_calls


def _physical_cpu_representatives() -> list[int]:
    """Return one allowed logical CPU per physical core where Linux exposes topology."""
    if not hasattr(os, "sched_getaffinity"):
        return list(range(os.cpu_count() or 1))

    allowed = sorted(os.sched_getaffinity(0))
    representatives: list[int] = []
    seen: set[tuple[str, str]] = set()

    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
            key = (package, core)
        except OSError:
            key = ("cpu", str(cpu))
        if key in seen:
            continue
        seen.add(key)
        representatives.append(cpu)

    return representatives or allowed


def _worker_affinity(dask_worker=None) -> dict[str, Any]:
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
    else:
        cpus = list(range(os.cpu_count() or 1))
    return {
        "address": getattr(dask_worker, "address", None),
        "name": str(getattr(dask_worker, "name", "unknown")),
        "cpus": cpus,
        "n_cpus": len(cpus),
    }


def _pin_worker_to_cpu(address_to_cpu: dict[str, int], dask_worker=None) -> dict[str, Any]:
    address = getattr(dask_worker, "address", None)
    if address not in address_to_cpu:
        raise RuntimeError(f"No CPU assignment for Dask worker {address!r}")
    cpu = int(address_to_cpu[address])
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU-affinity diagnostic requires os.sched_setaffinity().")
    os.sched_setaffinity(0, {cpu})
    return _worker_affinity(dask_worker=dask_worker)


def _median(results: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(result[key])
        for result in results
        if result.get("status") == "success" and result.get(key) is not None
    ]
    return float(np.median(values)) if values else None


def _run_point(args, *, workers: int, prepared, output: Path) -> None:
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
        info = client.scheduler_info().get("workers", {})
        addresses = sorted(info)
        if len(addresses) != workers:
            raise RuntimeError(f"Expected {workers} workers; observed {len(addresses)}")

        before = client.run(_worker_affinity)
        cpu_map: dict[str, int] = {}
        if args.cpu_affinity == "single":
            physical = _physical_cpu_representatives()
            if workers > len(physical):
                raise RuntimeError(
                    f"Requested {workers} pinned workers but only {len(physical)} "
                    "physical-core representatives are available in this CPU set."
                )
            cpu_map = {address: physical[index] for index, address in enumerate(addresses)}
            after = client.run(_pin_worker_to_cpu, address_to_cpu=cpu_map)
        else:
            after = client.run(_worker_affinity)

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
                results = execute_fold_calls(cvbench._bayesian_fold, calls, client=client)

            successes = [r for r in results if r.get("status") == "success"]
            failures = [r for r in results if r.get("status") != "success"]
            completed = len(successes)
            wall = float(timer["wall_seconds"])

            metadata = {
                "campaign": "bayesian-rsf-cpu-contention-v1",
                "sampler": args.sampler,
                "cpu_affinity": args.cpu_affinity,
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
                "median_prepare_seconds": _median(results, "prepare_seconds"),
                "median_model_seconds": _median(results, "model_seconds"),
                "median_min_ess_bulk": _median(results, "min_ess_bulk"),
                "median_max_rhat": _median(results, "max_rhat"),
                "total_divergences": int(
                    sum(int(r.get("n_divergences", 0)) for r in successes)
                ),
                "rows_per_individual": args.rows_per_individual,
                "predictors": args.predictors,
                "bin_width": args.bin_width,
                "random_slopes": args.random_slopes,
                "draws": args.draws,
                "tune": args.tune,
                "chains": args.chains,
                "cores_per_fold": args.cores,
                "affinity_before": before,
                "affinity_after": after,
                "cpu_map": cpu_map,
                "distinct_assigned_cpus": len(set(cpu_map.values())) if cpu_map else None,
                "failure_examples": [r.get("error") for r in failures[:3]],
            }

            append_benchmark_record(
                make_benchmark_record(
                    "bayesian_rsf_cpu_contention",
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
            print(
                f"sampler={args.sampler} affinity={args.cpu_affinity} "
                f"workers={workers} repeat={repeat} wall={wall:.3f}s "
                f"fold={metadata['median_fold_seconds']:.3f}s"
            )
    finally:
        client.close()
        cluster.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose Bayesian RSF outer-fold CPU contention with worker affinity."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=parse_positive_ints, default=[1, 2, 4, 6, 8, 12])
    parser.add_argument("--cpu-affinity", choices=("none", "single"), default="none")
    parser.add_argument("--folds", type=int, default=24)
    parser.add_argument("--rows-per-individual", type=int, default=500)
    parser.add_argument("--predictors", type=int, default=3)
    parser.add_argument("--sampler", default="blackjax")
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--random-slopes", type=int, default=1)
    parser.add_argument("--draws", type=int, default=250)
    parser.add_argument("--tune", type=int, default=250)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    args = parser.parse_args()

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

    for workers in args.workers:
        _run_point(args, workers=workers, prepared=prepared, output=args.output)


if __name__ == "__main__":
    main()
