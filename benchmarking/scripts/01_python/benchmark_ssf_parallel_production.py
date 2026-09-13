"""Benchmark the production per-individual SSF/iSSF parallel APIs.

Unlike the lighter concurrency sweep in ``benchmark_ssf_inference.py``, this
runner retains and returns the normal fitted model objects.  It therefore
captures process serialization/result-return overhead that matters to the
public production API.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hsa.compute import append_benchmark_record, benchmark_timer, make_benchmark_record
from hsa.compute.workloads import parse_positive_ints
from hsa.ssf.frequentist import fit_ssf_per_id
from hsa.ssf.issf import ISSFDesign
from hsa.ssf.parallel import fit_issf_per_id


def _synthetic_choices(
    *,
    n_strata: int,
    n_choices: int,
    predictors: int,
    individuals: int,
    analysis: str,
    seed: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    names = tuple(f"x{i}" for i in range(predictors))
    X = rng.normal(size=(n_strata, n_choices, predictors)).astype(
        np.float64,
        copy=False,
    )
    beta = np.linspace(0.4, -0.2, predictors, dtype=np.float64)
    offset = (
        rng.normal(scale=0.15, size=(n_strata, n_choices))
        if analysis == "issf"
        else np.zeros((n_strata, n_choices), dtype=np.float64)
    )
    eta = np.einsum("sjp,p->sj", X, beta, optimize=True) + offset
    probability = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
    cumulative = np.cumsum(probability, axis=1)
    chosen = np.sum(rng.random(n_strata)[:, None] > cumulative, axis=1)
    chosen = np.minimum(chosen, n_choices - 1)

    stratum = np.arange(n_strata, dtype=np.int64)
    individual = stratum % individuals
    used = np.zeros((n_strata, n_choices), dtype=np.int8)
    used[np.arange(n_strata), chosen] = 1

    data = {
        "id": np.repeat(individual, n_choices),
        "stratum_id": np.repeat(stratum, n_choices),
        "candidate_id": np.tile(np.arange(n_choices, dtype=np.int16), n_strata),
        "used": used.reshape(-1),
    }
    flat = X.reshape(-1, predictors)
    for index, name in enumerate(names):
        data[name] = flat[:, index]
    if analysis == "issf":
        data["proposal_offset"] = offset.reshape(-1)
    return pd.DataFrame(data), names


class _PreparedISSF:
    def __init__(self, design: ISSFDesign):
        self.id_col = design.id_col
        self._design = design

    def prepare_design(self, *, scaling=None, center_offset=True):
        return self._design


def _issf_design(
    frame: pd.DataFrame,
    predictors: tuple[str, ...],
    *,
    n_choices: int,
    individuals: int,
    n_strata: int,
) -> ISSFDesign:
    return ISSFDesign(
        data=frame,
        predictors=predictors,
        endpoint_predictors=predictors,
        start_predictors=(),
        directional_predictors=(),
        movement_terms=(),
        interaction_terms=(),
        interaction_columns={},
        scaling={},
        offset_col="proposal_offset",
        proposal_logpdf_col="proposal_logpdf",
        id_col="id",
        stratum_col="stratum_id",
        n_choices=n_choices,
        diagnostics={
            "n_individuals": individuals,
            "n_strata_retained": n_strata,
            "n_choice_rows": len(frame),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis", choices=("ssf", "issf"), required=True)
    parser.add_argument("--strata", type=int, default=100_000)
    parser.add_argument("--choices", type=int, default=11)
    parser.add_argument("--predictors", type=int, default=6)
    parser.add_argument("--individuals", type=int, default=20)
    parser.add_argument("--workers", type=parse_positive_ints, default=[1, 2, 4, 6, 8])
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--method", default="lbfgs")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if min(
        args.strata,
        args.choices,
        args.predictors,
        args.individuals,
        args.native_threads,
        args.maxiter,
        args.repeats,
    ) <= 0:
        parser.error("workload dimensions, maxiter, and repeats must be positive")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.append:
        raise SystemExit(f"{output} already exists; use --append intentionally")

    frame, names = _synthetic_choices(
        n_strata=args.strata,
        n_choices=args.choices,
        predictors=args.predictors,
        individuals=args.individuals,
        analysis=args.analysis,
        seed=args.seed,
    )
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    issf_analysis = None
    if args.analysis == "issf":
        issf_analysis = _PreparedISSF(
            _issf_design(
                frame,
                names,
                n_choices=args.choices,
                individuals=args.individuals,
                n_strata=args.strata,
            )
        )

    for repeat in range(1, args.repeats + 1):
        for workers in args.workers:
            with benchmark_timer() as timer:
                if args.analysis == "ssf":
                    fitted = fit_ssf_per_id(
                        frame,
                        id_col="id",
                        predictors=names,
                        reference_beta=np.zeros(len(names), dtype=np.float64),
                        engine="fast",
                        method=args.method,
                        maxiter=args.maxiter,
                        workers=workers,
                        executor=args.executor,
                        native_threads=args.native_threads,
                    )
                else:
                    fitted = fit_issf_per_id(
                        issf_analysis,
                        engine="fast",
                        method=args.method,
                        maxiter=args.maxiter,
                        workers=workers,
                        executor=args.executor,
                        native_threads=args.native_threads,
                    )

                # Touch returned result objects before ending the timed region so
                # lazy/result-transfer bugs cannot masquerade as fast fits.
                coefficient_checksum = float(
                    sum(
                        np.asarray(fit.result.params, dtype=float).sum()
                        for fit in fitted.fits.values()
                    )
                )
                n_fit_objects = len(fitted.fits)

            append_benchmark_record(
                make_benchmark_record(
                    f"inference_{args.analysis}_production_per_id",
                    timer["wall_seconds"],
                    rows=args.strata,
                    workers=min(workers, args.individuals),
                    threads_per_worker=args.native_threads,
                    bytes_processed=frame_bytes,
                    metadata={
                        "campaign": f"inference-{args.analysis}-production-per-id-v1",
                        "stage": "production_per_id_fit",
                        "analysis": args.analysis,
                        "executor": args.executor,
                        "workers_requested": workers,
                        "workers_effective": min(workers, args.individuals),
                        "native_threads": args.native_threads,
                        "n_strata": args.strata,
                        "n_choices": args.choices,
                        "predictors": args.predictors,
                        "individuals": args.individuals,
                        "choice_rows": len(frame),
                        "optimizer": args.method,
                        "repeat": repeat,
                        "fit_objects_retained": True,
                        "n_fit_objects": n_fit_objects,
                        "coefficient_checksum": coefficient_checksum,
                    },
                    operation_memory=timer,
                ),
                output,
            )
            del fitted
            gc.collect()


if __name__ == "__main__":
    main()
