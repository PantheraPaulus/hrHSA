"""Probe BlackJAX post-sampling diagnostics with explicit NumPy materialization.

This is intentionally a small validation harness rather than a general benchmark.
It samples one hierarchical RSF with PyMC's BlackJAX backend, reports the backing
array types returned by the backend, materializes only posterior/sample_stats to
host NumPy arrays, and then times hrHSA's existing ArviZ diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

from benchmark_inference import _synthetic_frame
from hsa.rsf.bayesian import (
    build_bayesian_rsf_model,
    evaluate_bayesian_rsf,
    prepare_bayesian_rsf_data,
)


def _backing_types(dataset) -> dict[str, str]:
    if dataset is None:
        return {}
    return {
        name: f"{type(value.data).__module__}.{type(value.data).__name__}"
        for name, value in dataset.data_vars.items()
    }


def _numpy_dataset(dataset):
    """Return an equivalent Xarray Dataset whose variables are host NumPy arrays."""
    if dataset is None:
        return None
    out = dataset.copy(deep=False)
    for name, value in dataset.data_vars.items():
        out[name] = value.copy(data=np.asarray(value.data))
    return out


def _tree_from_groups(*, posterior, sample_stats):
    """Build the minimal ArviZ/Xarray tree needed by hrHSA diagnostics.

    Recent ArviZ releases use :class:`xarray.DataTree` directly instead of the
    legacy ``arviz.InferenceData`` constructor.  Constructing the tree from
    datasets keeps this probe compatible with that migration while preserving
    the familiar ``idata.posterior`` / ``idata.sample_stats`` group access.
    """
    groups = {"posterior": posterior}
    if sample_stats is not None:
        groups["sample_stats"] = sample_stats
    return xr.DataTree.from_dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--predictors", type=int, default=3)
    parser.add_argument("--individuals", type=int, default=20)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--random-slopes", type=int, default=1)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=10001)
    args = parser.parse_args()

    import pymc as pm

    frame, names = _synthetic_frame(
        rows=args.rows,
        predictors=args.predictors,
        individuals=args.individuals,
        seed=args.seed,
    )
    data = prepare_bayesian_rsf_data(
        frame,
        names,
        id_col="individual-local-identifier",
        binning={name: args.bin_width for name in names},
    )
    model = build_bayesian_rsf_model(
        data,
        predictors=names,
        random_intercept=True,
        random_slopes=names[: args.random_slopes],
        store_eta=False,
    )

    sample_start = time.perf_counter()
    with model:
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            progressbar=False,
            return_inferencedata=True,
            random_seed=args.sampling_seed,
            compute_convergence_checks=False,
            nuts_sampler="blackjax",
        )
    sample_seconds = time.perf_counter() - sample_start

    before = {
        "posterior": _backing_types(idata.posterior),
        "sample_stats": _backing_types(getattr(idata, "sample_stats", None)),
    }

    materialize_start = time.perf_counter()
    posterior = _numpy_dataset(idata.posterior)
    sample_stats = _numpy_dataset(getattr(idata, "sample_stats", None))
    normalized = _tree_from_groups(
        posterior=posterior,
        sample_stats=sample_stats,
    )
    materialize_seconds = time.perf_counter() - materialize_start

    after = {
        "posterior": _backing_types(normalized.posterior),
        "sample_stats": _backing_types(getattr(normalized, "sample_stats", None)),
    }

    diagnostics_start = time.perf_counter()
    diagnostics = evaluate_bayesian_rsf(
        normalized,
        include_random_effects=False,
    )
    diagnostics_seconds = time.perf_counter() - diagnostics_start

    result = {
        "sample_seconds": sample_seconds,
        "materialize_seconds": materialize_seconds,
        "diagnostics_seconds": diagnostics_seconds,
        "total_after_sample_seconds": materialize_seconds + diagnostics_seconds,
        "backing_types_before": before,
        "backing_types_after": after,
        "max_rhat": diagnostics["max_rhat"],
        "min_ess_bulk": diagnostics["min_ess_bulk"],
        "min_ess_tail": diagnostics["min_ess_tail"],
        "n_divergences": diagnostics["n_divergences"],
        "max_tree_depth": diagnostics["max_tree_depth"],
        "mean_n_steps": diagnostics["mean_n_steps"],
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
