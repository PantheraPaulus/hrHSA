"""Benchmark SSF/iSSF frequentist and Bayesian inference scaling.

The synthetic workload preserves the canonical hrHSA choice-set invariant: every
stratum has a fixed number of alternatives and exactly one chosen alternative.
This lets the benchmark measure the optimized vectorized conditional likelihood
against Statsmodels on the same data, and measure the hierarchical Bayesian
categorical/softmax model without raster-I/O confounding.

The frequentist path additionally isolates the dense objective kernel and compares
the historical two-pass ``loglike + score`` evaluation against the fused production
kernel. Independent per-individual fits can be benchmarked with thread and process
executors so concurrency is measured before hrHSA commits to a public parallel-fit
API.

The SSF and iSSF modes share the same likelihood. ``analysis=issf`` adds canonical
movement-style predictors (``log_sl`` and ``cos_ta``) to the environmental design
so the benchmark exercises the integrated model shape used by iSSF fits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import inspect
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from hsa.compute import append_benchmark_record, benchmark_timer, make_benchmark_record
from hsa.compute.workloads import parse_positive_ints
from hsa.rsf.bayesian_hpc import _materialize_inference_tree
from hsa.ssf.bayesian import build_hierarchical_ssf_model
from hsa.ssf.data import build_ssf_choice_arrays
from hsa.ssf.fast_conditional import FastConditionalLogitModel
from hsa.ssf.frequentist import fit_conditional_ssf


def _predictor_names(analysis: str, predictors: int) -> list[str]:
    if predictors <= 0:
        raise ValueError("predictors must be positive")
    if analysis == "ssf":
        return [f"x{i}" for i in range(predictors)]
    if predictors < 3:
        raise ValueError("iSSF benchmark requires at least 3 predictors")
    environmental = [f"x{i}" for i in range(predictors - 2)]
    return [*environmental, "log_sl", "cos_ta"]


def _synthetic_choices(
    *,
    n_strata: int,
    n_choices: int,
    predictors: int,
    individuals: int,
    analysis: str,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    if min(n_strata, n_choices, predictors, individuals) <= 0:
        raise ValueError("all workload dimensions must be positive")
    if n_choices < 2:
        raise ValueError("choice sets need at least two alternatives")

    rng = np.random.default_rng(seed)
    names = _predictor_names(analysis, predictors)
    X = rng.normal(size=(n_strata, n_choices, len(names))).astype(
        np.float64,
        copy=False,
    )

    if analysis == "issf":
        log_sl_index = names.index("log_sl")
        cos_ta_index = names.index("cos_ta")
        X[:, :, log_sl_index] = rng.normal(
            0.0,
            0.7,
            size=(n_strata, n_choices),
        )
        X[:, :, cos_ta_index] = rng.uniform(
            -1.0,
            1.0,
            size=(n_strata, n_choices),
        )

    beta = np.linspace(0.45, -0.20, len(names), dtype=np.float64)
    eta = np.einsum("sjp,p->sj", X, beta, optimize=True)
    log_prob = eta - logsumexp(eta, axis=1, keepdims=True)
    probability = np.exp(log_prob)
    cumulative = np.cumsum(probability, axis=1)
    draws = rng.random(n_strata)
    chosen = np.sum(draws[:, None] > cumulative, axis=1)
    chosen = np.minimum(chosen, n_choices - 1).astype(np.int64)

    strata = np.arange(n_strata, dtype=np.int64)
    id_by_stratum = strata % individuals
    used = np.zeros((n_strata, n_choices), dtype=np.int8)
    used[np.arange(n_strata), chosen] = 1

    data: dict[str, Any] = {
        "id": np.repeat(id_by_stratum, n_choices),
        "stratum_id": np.repeat(strata, n_choices),
        "candidate_id": np.tile(np.arange(n_choices, dtype=np.int16), n_strata),
        "used": used.reshape(-1),
    }
    flat = X.reshape(-1, len(names))
    for index, name in enumerate(names):
        data[name] = flat[:, index]

    return pd.DataFrame(data), names


def _frame_bytes(frame: pd.DataFrame) -> int:
    return int(frame.memory_usage(index=True, deep=True).sum())


def _fit_metadata(result) -> dict[str, Any]:
    retvals = getattr(result, "mle_retvals", None) or {}
    return {
        "converged": bool(retvals.get("converged", getattr(result, "converged", True))),
        "iterations": retvals.get("iterations"),
        "optimizer_message": retvals.get("message"),
        "llf": float(result.llf),
    }


def _fit_individual(payload):
    frame, names, method, maxiter = payload
    _, result = fit_conditional_ssf(
        frame,
        predictors=names,
        id_col="id",
        stratum_col="stratum_id",
        engine="fast",
        method=method,
        maxiter=maxiter,
        disp=False,
    )
    return bool(getattr(result, "converged", True)), float(result.llf), int(len(frame))


def _fit_individual_process(payload):
    # Process workers do not share the parent's threadpoolctl context reliably.
    # Cap native linear-algebra pools inside each process explicitly.
    with threadpool_limits(limits=1):
        return _fit_individual(payload)


def _bayesian_diagnostics(idata, completed_seconds: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import arviz as az

        summary = az.summary(idata, var_names=["mu_beta"], round_to=None)
        if "ess_bulk" in summary:
            ess = summary["ess_bulk"].to_numpy(dtype=float)
            out["ess_bulk_min"] = float(np.nanmin(ess))
            out["ess_bulk_median"] = float(np.nanmedian(ess))
            if completed_seconds > 0:
                out["ess_bulk_min_per_second"] = out["ess_bulk_min"] / completed_seconds
        if "r_hat" in summary:
            out["r_hat_max"] = float(
                np.nanmax(summary["r_hat"].to_numpy(dtype=float))
            )
    except Exception as exc:
        out["diagnostics_error"] = repr(exc)

    try:
        diverging = np.asarray(idata.sample_stats["diverging"])
        out["divergences"] = int(diverging.sum())
    except Exception:
        pass
    return out


def _benchmark_objective_kernel(
    *,
    frame: pd.DataFrame,
    names: list[str],
    args,
    n_strata: int,
    repeat: int,
    output: Path,
) -> None:
    with threadpool_limits(limits=1):
        arrays = build_ssf_choice_arrays(
            frame,
            id_col="id",
            predictors=names,
            stratum_col="stratum_id",
            candidate_col="candidate_id",
            used_col="used",
            dtype="float64",
        )
        model = FastConditionalLogitModel(arrays)
        beta = np.linspace(-0.20, 0.20, model.n_predictors, dtype=np.float64)
        evaluations = int(args.objective_evaluations)

        with benchmark_timer() as separate_timer:
            for _ in range(evaluations):
                model.loglike(beta)
                model.score(beta)

        with benchmark_timer() as fused_timer:
            for _ in range(evaluations):
                model.loglike_and_score(beta)

    for kernel, timer in (
        ("separate", separate_timer),
        ("fused", fused_timer),
    ):
        append_benchmark_record(
            make_benchmark_record(
                f"inference_{args.analysis}_objective_kernel",
                timer["wall_seconds"],
                rows=len(frame) * evaluations,
                workers=1,
                threads_per_worker=1,
                bytes_processed=int(arrays.X.nbytes) * evaluations,
                metadata={
                    "campaign": f"inference-{args.analysis}-frequentist-v2",
                    "stage": "objective_kernel",
                    "analysis": args.analysis,
                    "kernel": kernel,
                    "n_strata": n_strata,
                    "n_choices": args.choices,
                    "predictors": len(names),
                    "individuals": args.individuals,
                    "evaluations": evaluations,
                    "repeat": repeat,
                    "array_bytes": int(arrays.X.nbytes),
                },
                operation_memory=timer,
            ),
            output,
        )


def _benchmark_per_individual(
    *,
    frame: pd.DataFrame,
    names: list[str],
    args,
    n_strata: int,
    output: Path,
) -> None:
    if args.skip_per_id or n_strata > args.per_id_max_strata:
        return

    with benchmark_timer() as split_timer:
        groups = [
            group.copy()
            for _, group in frame.groupby("id", sort=False, observed=True)
        ]
    append_benchmark_record(
        make_benchmark_record(
            f"inference_{args.analysis}_per_id_split",
            split_timer["wall_seconds"],
            rows=len(frame),
            workers=1,
            threads_per_worker=1,
            bytes_processed=_frame_bytes(frame),
            metadata={
                "campaign": f"inference-{args.analysis}-frequentist-v2",
                "stage": "per_id_split",
                "analysis": args.analysis,
                "n_strata": n_strata,
                "individuals": len(groups),
            },
            operation_memory=split_timer,
        ),
        output,
    )

    payloads = [
        (group, names, args.per_id_method, args.maxiter)
        for group in groups
    ]
    executors = [item.strip() for item in args.per_id_executors.split(",") if item.strip()]
    unknown = sorted(set(executors).difference({"thread", "process"}))
    if unknown:
        raise ValueError(f"Unknown per-ID executors: {unknown}")

    for executor_name in executors:
        for workers in args.per_id_workers:
            if workers > len(groups):
                continue
            for repeat in range(1, args.repeats + 1):
                if executor_name == "thread":
                    with threadpool_limits(limits=1):
                        with benchmark_timer() as timer:
                            with ThreadPoolExecutor(max_workers=workers) as pool:
                                results = list(pool.map(_fit_individual, payloads))
                else:
                    with benchmark_timer() as timer:
                        with ProcessPoolExecutor(max_workers=workers) as pool:
                            results = list(pool.map(_fit_individual_process, payloads))

                if not all(result[0] for result in results):
                    raise RuntimeError("At least one per-individual fit did not converge.")
                append_benchmark_record(
                    make_benchmark_record(
                        f"inference_{args.analysis}_per_id_fit",
                        timer["wall_seconds"],
                        rows=len(groups),
                        workers=workers,
                        threads_per_worker=1,
                        bytes_processed=_frame_bytes(frame),
                        metadata={
                            "campaign": f"inference-{args.analysis}-frequentist-v2",
                            "stage": "per_id_fit",
                            "analysis": args.analysis,
                            "executor": executor_name,
                            "workers": workers,
                            "n_strata": n_strata,
                            "n_choices": args.choices,
                            "predictors": len(names),
                            "individuals": len(groups),
                            "choice_rows": int(len(frame)),
                            "optimizer": args.per_id_method,
                            "repeat": repeat,
                        },
                        operation_memory=timer,
                    ),
                    output,
                )


def _run_frequentist(args) -> None:
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.append:
        raise SystemExit(f"{output} already exists; use --append intentionally")

    engines = [item.strip() for item in args.engines.split(",") if item.strip()]
    if not engines or any(item not in {"fast", "statsmodels"} for item in engines):
        raise ValueError("engines must contain fast and/or statsmodels")

    for n_strata in args.strata:
        frame, names = _synthetic_choices(
            n_strata=n_strata,
            n_choices=args.choices,
            predictors=args.predictors,
            individuals=args.individuals,
            analysis=args.analysis,
            seed=args.seed + n_strata,
        )
        frame_bytes = _frame_bytes(frame)

        for repeat in range(1, args.repeats + 1):
            _benchmark_objective_kernel(
                frame=frame,
                names=names,
                args=args,
                n_strata=n_strata,
                repeat=repeat,
                output=output,
            )

            for threads in args.blas_threads:
                with threadpool_limits(limits=threads):
                    with benchmark_timer() as timer:
                        arrays = build_ssf_choice_arrays(
                            frame,
                            id_col="id",
                            predictors=names,
                            stratum_col="stratum_id",
                            candidate_col="candidate_id",
                            used_col="used",
                            dtype="float64",
                        )
                    append_benchmark_record(
                        make_benchmark_record(
                            f"inference_{args.analysis}_choice_array_prepare",
                            timer["wall_seconds"],
                            rows=len(frame),
                            workers=1,
                            threads_per_worker=threads,
                            bytes_processed=frame_bytes,
                            metadata={
                                "campaign": f"inference-{args.analysis}-frequentist-v2",
                                "stage": "choice_array_prepare",
                                "analysis": args.analysis,
                                "n_strata": n_strata,
                                "n_choices": args.choices,
                                "predictors": len(names),
                                "individuals": args.individuals,
                                "blas_threads": threads,
                                "repeat": repeat,
                                "array_bytes": int(arrays.X.nbytes),
                            },
                            operation_memory=timer,
                        ),
                        output,
                    )
                    del arrays

                    reference = None
                    for engine in engines:
                        if engine == "statsmodels" and n_strata > args.statsmodels_max_strata:
                            continue
                        with benchmark_timer() as timer:
                            _, result = fit_conditional_ssf(
                                frame,
                                predictors=names,
                                id_col="id",
                                stratum_col="stratum_id",
                                engine=engine,
                                method=args.method,
                                maxiter=args.maxiter,
                                disp=False,
                            )
                        params = np.asarray(result.params, dtype=float)
                        if reference is None:
                            reference = params.copy()
                        coefficient_difference = float(
                            np.max(np.abs(params - reference))
                        )
                        append_benchmark_record(
                            make_benchmark_record(
                                f"inference_{args.analysis}_frequentist_fit",
                                timer["wall_seconds"],
                                rows=len(frame),
                                workers=1,
                                threads_per_worker=threads,
                                bytes_processed=frame_bytes,
                                metadata={
                                    "campaign": f"inference-{args.analysis}-frequentist-v2",
                                    "stage": "optimizer",
                                    "analysis": args.analysis,
                                    "engine": engine,
                                    "optimizer": args.method,
                                    "n_strata": n_strata,
                                    "n_choices": args.choices,
                                    "predictors": len(names),
                                    "individuals": args.individuals,
                                    "blas_threads": threads,
                                    "repeat": repeat,
                                    "max_abs_coef_diff_vs_first_engine": coefficient_difference,
                                    **_fit_metadata(result),
                                },
                                operation_memory=timer,
                            ),
                            output,
                        )

        _benchmark_per_individual(
            frame=frame,
            names=names,
            args=args,
            n_strata=n_strata,
            output=output,
        )

        del frame
        import gc
        gc.collect()


def _run_bayesian(args) -> None:
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.append:
        raise SystemExit(f"{output} already exists; use --append intentionally")

    try:
        import pymc as pm
    except ImportError as exc:
        raise SystemExit("Bayesian SSF benchmark requires hsa[bayesian].") from exc

    samplers = [item.strip() for item in args.samplers.split(",") if item.strip()]
    signature = inspect.signature(pm.sample)

    for n_strata in args.strata:
        frame, names = _synthetic_choices(
            n_strata=n_strata,
            n_choices=args.choices,
            predictors=args.predictors,
            individuals=args.individuals,
            analysis=args.analysis,
            seed=args.seed + n_strata,
        )
        frame_bytes = _frame_bytes(frame)

        for repeat in range(1, args.repeats + 1):
            with benchmark_timer() as timer:
                arrays = build_ssf_choice_arrays(
                    frame,
                    id_col="id",
                    predictors=names,
                    stratum_col="stratum_id",
                    candidate_col="candidate_id",
                    used_col="used",
                    dtype="float32",
                )
            append_benchmark_record(
                make_benchmark_record(
                    f"inference_{args.analysis}_bayesian_prepare",
                    timer["wall_seconds"],
                    rows=len(frame),
                    bytes_processed=frame_bytes,
                    metadata={
                        "campaign": f"inference-{args.analysis}-bayesian-v1",
                        "stage": "choice_array_prepare",
                        "analysis": args.analysis,
                        "n_strata": n_strata,
                        "n_choices": args.choices,
                        "predictors": len(names),
                        "individuals": args.individuals,
                        "repeat": repeat,
                        "array_bytes": int(arrays.X.nbytes),
                    },
                    operation_memory=timer,
                ),
                output,
            )

            for sampler in samplers:
                with benchmark_timer() as timer:
                    model = build_hierarchical_ssf_model(arrays)
                append_benchmark_record(
                    make_benchmark_record(
                        f"inference_{args.analysis}_bayesian_model_build",
                        timer["wall_seconds"],
                        rows=n_strata,
                        metadata={
                            "campaign": f"inference-{args.analysis}-bayesian-v1",
                            "stage": "model_build",
                            "analysis": args.analysis,
                            "sampler": sampler,
                            "n_strata": n_strata,
                            "n_choices": args.choices,
                            "predictors": len(names),
                            "individuals": args.individuals,
                            "repeat": repeat,
                        },
                        operation_memory=timer,
                    ),
                    output,
                )

                kwargs: dict[str, Any] = {
                    "draws": args.draws,
                    "tune": args.tune,
                    "chains": args.chains,
                    "cores": args.cores,
                    "target_accept": args.target_accept,
                    "progressbar": False,
                    "return_inferencedata": True,
                    "random_seed": args.seed + repeat,
                }
                if "compute_convergence_checks" in signature.parameters:
                    kwargs["compute_convergence_checks"] = False
                if "nuts_sampler" in signature.parameters:
                    kwargs["nuts_sampler"] = sampler
                elif sampler != "pymc":
                    print(
                        f"Skipping sampler={sampler}: this PyMC version has no "
                        "nuts_sampler argument."
                    )
                    continue

                try:
                    with benchmark_timer() as timer:
                        dispatch_started = perf_counter()
                        with model:
                            raw_idata = pm.sample(**kwargs)
                        dispatch_seconds = perf_counter() - dispatch_started
                        materialize_started = perf_counter()
                        idata = _materialize_inference_tree(raw_idata)
                        materialize_seconds = perf_counter() - materialize_started
                    completed = float(timer["wall_seconds"])
                    diagnostics = _bayesian_diagnostics(idata, completed)
                    append_benchmark_record(
                        make_benchmark_record(
                            f"inference_{args.analysis}_bayesian_sample",
                            completed,
                            rows=n_strata,
                            workers=args.chains,
                            threads_per_worker=max(1, args.cores // args.chains),
                            metadata={
                                "campaign": f"inference-{args.analysis}-bayesian-v1",
                                "stage": "completed_sampling",
                                "status": "success",
                                "analysis": args.analysis,
                                "sampler": sampler,
                                "n_strata": n_strata,
                                "n_choices": args.choices,
                                "predictors": len(names),
                                "individuals": args.individuals,
                                "draws": args.draws,
                                "tune": args.tune,
                                "chains": args.chains,
                                "cores": args.cores,
                                "target_accept": args.target_accept,
                                "repeat": repeat,
                                "dispatch_seconds": dispatch_seconds,
                                "materialize_seconds": materialize_seconds,
                                **diagnostics,
                            },
                            operation_memory=timer,
                        ),
                        output,
                    )
                    del raw_idata, idata
                except Exception as exc:
                    append_benchmark_record(
                        make_benchmark_record(
                            f"inference_{args.analysis}_bayesian_failure",
                            0.0,
                            rows=n_strata,
                            workers=args.chains,
                            metadata={
                                "campaign": f"inference-{args.analysis}-bayesian-v1",
                                "stage": "completed_sampling",
                                "status": "failed",
                                "analysis": args.analysis,
                                "sampler": sampler,
                                "n_strata": n_strata,
                                "n_choices": args.choices,
                                "predictors": len(names),
                                "individuals": args.individuals,
                                "repeat": repeat,
                                "exception_type": type(exc).__name__,
                                "exception_message": str(exc)[:1000],
                            },
                        ),
                        output,
                    )
                    print(f"Bayesian run failed for {sampler}: {exc}")

            del arrays, model

        del frame
        import gc
        gc.collect()


def _common(subparser) -> None:
    subparser.add_argument("--output", type=Path, required=True)
    subparser.add_argument("--analysis", choices=("ssf", "issf"), default="ssf")
    subparser.add_argument("--strata", type=parse_positive_ints, required=True)
    subparser.add_argument("--choices", type=int, default=20)
    subparser.add_argument("--predictors", type=int, default=6)
    subparser.add_argument("--individuals", type=int, default=20)
    subparser.add_argument("--repeats", type=int, default=3)
    subparser.add_argument("--seed", type=int, default=42)
    subparser.add_argument("--append", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    frequentist = sub.add_parser("frequentist")
    _common(frequentist)
    frequentist.add_argument("--engines", default="fast,statsmodels")
    frequentist.add_argument("--statsmodels-max-strata", type=int, default=50_000)
    frequentist.add_argument("--method", default="bfgs")
    frequentist.add_argument("--maxiter", type=int, default=300)
    frequentist.add_argument("--blas-threads", type=parse_positive_ints, default=[1])
    frequentist.add_argument("--objective-evaluations", type=int, default=20)
    frequentist.add_argument("--per-id-workers", type=parse_positive_ints, default=[1, 2, 4])
    frequentist.add_argument("--per-id-executors", default="thread")
    frequentist.add_argument("--per-id-method", default="lbfgs")
    frequentist.add_argument("--per-id-max-strata", type=int, default=250_000)
    frequentist.add_argument("--skip-per-id", action="store_true")

    bayesian = sub.add_parser("bayesian")
    _common(bayesian)
    bayesian.add_argument("--samplers", default="pymc,blackjax")
    bayesian.add_argument("--draws", type=int, default=1000)
    bayesian.add_argument("--tune", type=int, default=1000)
    bayesian.add_argument("--chains", type=int, default=4)
    bayesian.add_argument("--cores", type=int, default=4)
    bayesian.add_argument("--target-accept", type=float, default=0.9)

    args = parser.parse_args()
    if min(args.choices, args.predictors, args.individuals, args.repeats) <= 0:
        parser.error("workload dimensions and repeats must be positive")
    if args.command == "frequentist":
        if min(
            args.statsmodels_max_strata,
            args.maxiter,
            args.objective_evaluations,
            args.per_id_max_strata,
        ) <= 0:
            parser.error("frequentist benchmark controls must be positive")
        _run_frequentist(args)
    else:
        if min(args.draws, args.tune, args.chains, args.cores) <= 0:
            parser.error("draws/tune/chains/cores must be positive")
        _run_bayesian(args)


if __name__ == "__main__":
    main()
