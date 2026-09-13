"""Benchmark vectorized BlackJAX HMC for hierarchical Bayesian RSF folds.

This is a mechanism diagnostic, not a production sampler switch. It keeps the
existing PyMC-built RSF target but replaces NUTS' dynamic trajectory with static
HMC and runs the chains through ``jax.vmap``. The benchmark includes PyMC/JAX
initialisation, BlackJAX window adaptation, sampling, posterior post-processing,
explicit materialisation, diagnostics, and outer-fold Dask concurrency.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as package_version
import os
from functools import partial
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


def _hmc_info_stats(state, info) -> dict[str, Any]:
    """Return sampler statistics that exist for static HMC."""
    return {
        "diverging": info.is_divergent,
        "energy": info.energy,
        "n_steps": info.num_integration_steps,
        "acceptance_rate": info.acceptance_rate,
        "lp": state.logdensity,
    }


def _blackjax_hmc_inference_loop(
    seed,
    init_position,
    *,
    logp_fn,
    draws: int,
    tune: int,
    target_accept: float,
    integration_steps: int,
):
    """Warm up and sample one static-HMC chain.

    This mirrors PyMC 6.2's private BlackJAX loop, except that its NUTS-only
    ``num_trajectory_expansions`` statistic is deliberately not accessed.
    """
    import blackjax
    import jax

    from blackjax.adaptation.base import get_filter_adapt_info_fn

    warmup_key, sample_key = jax.random.split(seed)
    adapt = blackjax.window_adaptation(
        algorithm=blackjax.hmc,
        logdensity_fn=logp_fn,
        target_acceptance_rate=target_accept,
        adaptation_info_fn=get_filter_adapt_info_fn(),
        num_integration_steps=integration_steps,
    )
    (last_state, tuned_params), _ = adapt.run(
        warmup_key,
        init_position,
        num_steps=tune,
    )
    kernel = blackjax.hmc(logp_fn, **tuned_params).step

    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state.position, _hmc_info_stats(state, info))

    keys = jax.random.split(sample_key, draws)
    _, (positions, stats) = jax.lax.scan(one_step, last_state, keys)
    return positions, stats, tuned_params["step_size"]


def _sample_blackjax_hmc_vectorized(
    model,
    *,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    integration_steps: int,
    seed: int,
):
    """Sample a PyMC model with SIMD/static HMC and return a DataTree."""
    import blackjax
    import jax
    import pymc.sampling.jax as pmjax

    from arviz_base import from_dict, make_attrs
    from pymc.backends.arviz import coords_and_dims_for_inferencedata
    from pymc.util import get_default_varnames

    logp_fn = pmjax.get_jaxified_logp(model)
    initial_points = pmjax._get_batched_jittered_initial_points(
        model=model,
        chains=chains,
        initvals=None,
        random_seed=seed,
        jitter=True,
        logp_fn=logp_fn,
    )
    if chains == 1:
        initial_points = [np.expand_dims(value, axis=0) for value in initial_points]

    keys = jax.random.split(jax.random.PRNGKey(seed), chains)
    run_chain = partial(
        _blackjax_hmc_inference_loop,
        logp_fn=logp_fn,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        integration_steps=integration_steps,
    )

    raw_mcmc_samples, sample_stats, tuned_step_sizes = jax.vmap(run_chain)(
        keys,
        initial_points,
    )

    vars_to_sample = list(
        get_default_varnames(model.unobserved_value_vars, include_transformed=False)
    )
    jax_fn = pmjax.get_jaxified_graph(
        inputs=model.value_vars,
        outputs=vars_to_sample,
    )
    posterior_values = pmjax._postprocess_samples(
        jax_fn,
        raw_mcmc_samples,
        postprocessing_backend=None,
        postprocessing_vectorize="vmap",
        donate_samples=True,
    )
    posterior = {
        variable.name: values
        for variable, values in zip(vars_to_sample, posterior_values)
    }

    coords, dims = coords_and_dims_for_inferencedata(model)
    attrs = {
        "posterior": make_attrs(
            {
                "tuning_steps": tune,
                "algorithm": "blackjax_hmc_vectorized",
                "integration_steps": integration_steps,
            },
            inference_library=blackjax,
        )
    }
    idata = from_dict(
        data={
            "posterior": posterior,
            "sample_stats": sample_stats,
        },
        coords=coords,
        dims=dims,
        sample_dims=["chain", "draw"],
        attrs=attrs,
    )
    return idata, tuned_step_sizes


def _hmc_fold(
    *,
    prepared_root: str,
    heldout_id: Any,
    predictors: list[str],
    bin_width: float,
    random_slopes: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    integration_steps: int,
    seed: int,
) -> dict[str, Any]:
    global _WORKER_FOLD_SEQUENCE
    _WORKER_FOLD_SEQUENCE += 1
    sequence = _WORKER_FOLD_SEQUENCE
    started = perf_counter()

    worker_address = None
    worker_name = None
    stage = "worker_setup"

    try:
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
        raw_idata, tuned_step_sizes = _sample_blackjax_hmc_vectorized(
            model,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            integration_steps=integration_steps,
            seed=seed,
        )
        dispatch_seconds = perf_counter() - sample_started

        stage = "materialize"
        materialize_started = perf_counter()
        idata = _materialize_inference_tree(raw_idata)
        tuned_step_sizes = np.asarray(tuned_step_sizes)
        materialize_seconds = perf_counter() - materialize_started
        completed_sampling_seconds = dispatch_seconds + materialize_seconds
        del raw_idata

        stage = "diagnostics"
        diagnostics_started = perf_counter()
        diagnostics = _sampling_diagnostics(idata)
        diagnostics_seconds = perf_counter() - diagnostics_started

        stats = idata.sample_stats
        mean_acceptance_rate = float(
            np.asarray(stats["acceptance_rate"], dtype=float).mean()
        )
        mean_n_steps = float(np.asarray(stats["n_steps"], dtype=float).mean())
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
            "mean_acceptance_rate": mean_acceptance_rate,
            "mean_n_steps": mean_n_steps,
            "mean_tuned_step_size": float(tuned_step_sizes.mean()),
            "worker_address": worker_address,
            "worker_name": worker_name,
            "worker_pid": os.getpid(),
            "worker_fold_sequence": sequence,
            "pymc_version": _distribution_version("pymc"),
            "jax_version": _distribution_version("jax"),
            "blackjax_version": _distribution_version("blackjax"),
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
            "pymc_version": _distribution_version("pymc"),
            "jax_version": _distribution_version("jax"),
            "blackjax_version": _distribution_version("blackjax"),
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


def _run_point(args, *, workers: int, integration_steps: int, prepared, output: Path) -> None:
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
                    "target_accept": args.target_accept,
                    "integration_steps": integration_steps,
                    "seed": args.seed + repeat + 100_000 * fold_index,
                }
                for fold_index, heldout_id in enumerate(ids)
            ]

            with benchmark_timer(client=client) as timer:
                results = execute_fold_calls(_hmc_fold, calls, client=client)

            successes = [r for r in results if r.get("status") == "success"]
            failures = [r for r in results if r.get("status") != "success"]
            completed = len(successes)
            wall = float(timer["wall_seconds"])
            metadata = {
                "campaign": "bayesian-rsf-hmc-vectorized-v1",
                "sampler": "blackjax",
                "algorithm": "hmc",
                "chain_method": "vectorized",
                "workers": workers,
                "repeat": repeat,
                "folds_requested": len(ids),
                "folds_completed": completed,
                "folds_failed": len(failures),
                "folds_per_second": completed / wall if wall > 0 else np.nan,
                "integration_steps": integration_steps,
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
                "median_mean_acceptance_rate": _median(
                    results, "mean_acceptance_rate"
                ),
                "median_mean_n_steps": _median(results, "mean_n_steps"),
                "median_mean_tuned_step_size": _median(
                    results, "mean_tuned_step_size"
                ),
                "median_min_ess_bulk": _median(results, "min_ess_bulk"),
                "median_min_ess_tail": _median(results, "min_ess_tail"),
                "median_max_rhat": _median(results, "max_rhat"),
                "total_divergences": int(
                    sum(int(r.get("n_divergences", 0)) for r in successes)
                ),
                "sequence_completed_sampling_medians": _sequence_medians(results),
                "rows_per_individual": args.rows_per_individual,
                "predictors": args.predictors,
                "bin_width": args.bin_width,
                "random_slopes": args.random_slopes,
                "pymc_versions": sorted(
                    {str(r["pymc_version"]) for r in results if r.get("pymc_version")}
                ),
                "jax_versions": sorted(
                    {str(r["jax_version"]) for r in results if r.get("jax_version")}
                ),
                "blackjax_versions": sorted(
                    {
                        str(r["blackjax_version"])
                        for r in results
                        if r.get("blackjax_version")
                    }
                ),
                "failure_examples": [
                    f"{r.get('stage')}: {r.get('error')}" for r in failures[:3]
                ],
                "observed_workers": observed,
            }

            append_benchmark_record(
                make_benchmark_record(
                    "bayesian_rsf_hmc_vectorized",
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
                f"hmc-vmap: workers={workers} L={integration_steps} "
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
        description=(
            "Benchmark four-chain vectorized BlackJAX static HMC on Bayesian RSF folds."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=parse_positive_ints, default=[4, 12])
    parser.add_argument(
        "--integration-steps",
        type=parse_positive_ints,
        default=[32, 64],
        help="Comma-separated fixed HMC leapfrog counts; default: 32,64.",
    )
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

    for name in (
        "folds",
        "rows_per_individual",
        "predictors",
        "draws",
        "tune",
        "chains",
        "repeats",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.random_slopes < 0 or args.random_slopes > args.predictors:
        parser.error("--random-slopes must be between 0 and --predictors")
    if not 0.0 < args.target_accept < 1.0:
        parser.error("--target-accept must be between 0 and 1")

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

    for integration_steps in args.integration_steps:
        for workers in args.workers:
            _run_point(
                args,
                workers=workers,
                integration_steps=integration_steps,
                prepared=prepared,
                output=args.output,
            )


if __name__ == "__main__":
    main()
