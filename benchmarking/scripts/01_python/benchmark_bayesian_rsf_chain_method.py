"""Compatibility entry point for the corrected Bayesian RSF chain benchmark."""

from __future__ import annotations

import inspect
from typing import Any

from benchmark_bayesian_rsf_chain_method_impl import (
    _diagnostic_fold,
    _distinct_failure_examples,
    _format_seconds,
    _median,
    _post_sampling_jax_runtime,
    _run_point,
    _sequence_medians,
    main,
)


def _configure_sampling(
    pm,
    *,
    sampler: str,
    chain_method: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    seed: int,
) -> tuple[dict[str, Any], str]:
    """Reconstruct the historical/explicit ``pm.sample`` keyword geometry.

    This helper is retained for benchmark regression tests and old diagnostic
    notebooks. The corrected runtime implementation in
    ``benchmark_bayesian_rsf_chain_method_impl`` uses PyMC's direct JAX driver
    for explicit chain methods so JAX is not initialized before PyMC configures
    the CPU device geometry.
    """
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

    if chain_method == "default":
        return sampling, "historical_default"

    sampling["nuts"] = {"chain_method": chain_method}
    return sampling, "explicit_chain_method"


if __name__ == "__main__":
    main()
