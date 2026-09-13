"""Probe PyMC's JAX RSF chain driver without pm.sample routing.

This is a mechanism diagnostic, not a production benchmark. PyMC 6.0--6.2 does
not forward ``chain_method`` from ``pm.sample(..., nuts={...})`` to
``sample_jax_nuts``. Calling the JAX driver directly lets us distinguish that
routing limitation from failures in JAX's parallel/vectorized chain execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

import benchmark_cv_scaling as cvbench
from hsa.rsf.bayesian import build_bayesian_rsf_model, prepare_bayesian_rsf_data
from hsa.rsf.bayesian_hpc import _materialize_inference_tree


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe BlackJAX RSF chain_method through sample_jax_nuts directly."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--chain-method",
        choices=("parallel", "vectorized"),
        default="vectorized",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--rows-per-individual", type=int, default=500)
    parser.add_argument("--predictors", type=int, default=3)
    parser.add_argument("--random-slopes", type=int, default=1)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--draws", type=int, default=25)
    parser.add_argument("--tune", type=int, default=25)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()

    if args.random_slopes < 0 or args.random_slopes > args.predictors:
        parser.error("--random-slopes must be between 0 and --predictors")

    args.work_dir = args.work_dir.expanduser().resolve()
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

    heldout_id = prepared.individuals[0]
    predictors = list(prepared.predictors)
    train_df = prepared.load(exclude=heldout_id)
    bayes_data = prepare_bayesian_rsf_data(
        train_df,
        predictors,
        id_col="individual-local-identifier",
        binning={name: args.bin_width for name in predictors},
    )
    model = build_bayesian_rsf_model(
        bayes_data,
        predictors=predictors,
        random_intercept=True,
        random_slopes=predictors[: args.random_slopes],
        store_eta=False,
    )

    import blackjax
    import jax
    import pymc as pm
    from pymc.sampling.jax import sample_jax_nuts

    metadata = {
        "pymc_version": str(pm.__version__),
        "jax_version": str(jax.__version__),
        "blackjax_version": str(blackjax.__version__),
        "jax_local_device_count": int(jax.local_device_count()),
        "jax_devices": [str(device) for device in jax.local_devices()],
        "chain_method": args.chain_method,
        "chains": args.chains,
        "draws": args.draws,
        "tune": args.tune,
        "heldout_id": str(heldout_id),
        "status": "failed",
    }

    started = perf_counter()
    try:
        with model:
            raw_idata = sample_jax_nuts(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                target_accept=args.target_accept,
                random_seed=args.seed + 1,
                model=model,
                progressbar=False,
                quiet=True,
                chain_method=args.chain_method,
                nuts_sampler="blackjax",
                compute_convergence_checks=False,
            )
        dispatch_seconds = perf_counter() - started

        materialize_started = perf_counter()
        idata = _materialize_inference_tree(raw_idata)
        materialize_seconds = perf_counter() - materialize_started

        metadata.update(
            status="success",
            dispatch_seconds=float(dispatch_seconds),
            materialize_seconds=float(materialize_seconds),
            completed_sampling_seconds=float(dispatch_seconds + materialize_seconds),
            posterior_variables=sorted(str(name) for name in idata.posterior.data_vars),
        )
    except Exception as exc:
        metadata.update(
            elapsed_seconds=float(perf_counter() - started),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )

    print(json.dumps(metadata, indent=2, sort_keys=True))
    if metadata["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
