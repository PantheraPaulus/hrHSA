"""CPU execution planning for Bayesian model fitting.

The planner separates two different performance objectives:

* latency: finish one posterior as quickly and efficiently as possible;
* throughput: finish a collection of independent posteriors as quickly as possible.

The recommendations encode benchmarked CPU behavior, but do not silently change a
sampler in the scientific fitting APIs. Callers can inspect the returned plan and
pass its sampler/geometry explicitly to PyMC, Dask, or a batch launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_ANALYSES = {"rsf", "ssf", "issf"}
_OBJECTIVES = {"auto", "latency", "throughput"}
_PROFILES = {"generic", "coolmuc4"}


@dataclass(frozen=True)
class BayesianCPUPlan:
    """Explicit CPU geometry for one Bayesian workload.

    ``outer_workers`` is the number of independent models/folds to execute at
    once. ``cores_per_fit`` is the maximum number of CPU cores assigned to one
    fit, normally used for concurrent chains by CPU-native samplers such as
    nutpie. Four chains remain the statistical default even when
    ``cores_per_fit`` is one; in that case the chains execute sequentially.

    ``active_cores`` is the requested worker geometry, not a claim about hidden
    runtime threads used by JAX/XLA.
    """

    analysis: str
    objective: str
    sampler: str
    n_models: int
    available_cores: int
    chains: int
    outer_workers: int
    cores_per_fit: int
    active_cores: int
    unused_cores: int
    profile: str = "generic"

    @property
    def parallel_models(self) -> int:
        """Alias for the outer model/fold concurrency."""
        return self.outer_workers

    def pymc_sample_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return portable PyMC sampling controls for this plan."""
        options: dict[str, Any] = {
            "nuts_sampler": self.sampler,
            "chains": self.chains,
            "cores": self.cores_per_fit,
        }
        options.update(overrides)
        return options


def plan_bayesian_cpu_execution(
    *,
    analysis: str,
    n_models: int,
    available_cores: int,
    chains: int = 4,
    objective: str = "auto",
    profile: str = "generic",
) -> BayesianCPUPlan:
    """Plan a measured CPU-oriented Bayesian execution geometry.

    ``profile='generic'`` preserves the portable outer-first rule: independent
    fits are filled first and spare cores are assigned to concurrent chains.

    ``profile='coolmuc4'`` applies the measured one-node CoolMUC-4 throughput
    policy when exactly 112 physical cores and at least four chains are available:

    * RSF: cap outer concurrency at 56 fits, yielding 56x2 for large ensembles;
    * SSF/iSSF: cap outer concurrency at 28 fits, yielding 28x4 for large ensembles.

    Smaller allocations or fewer than four chains fall back to the portable rule.
    The site-specific policy is deliberately opt-in rather than a hidden default.
    """

    analysis = str(analysis).lower()
    objective = str(objective).lower()
    profile = str(profile).lower()
    if analysis not in _ANALYSES:
        raise ValueError(f"analysis must be one of {sorted(_ANALYSES)}")
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_OBJECTIVES)}")
    if profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}")
    if n_models <= 0:
        raise ValueError("n_models must be positive")
    if available_cores <= 0:
        raise ValueError("available_cores must be positive")
    if chains <= 0:
        raise ValueError("chains must be positive")

    resolved_objective = (
        "latency"
        if objective == "auto" and n_models == 1
        else "throughput"
        if objective == "auto"
        else objective
    )

    if resolved_objective == "latency":
        outer_workers = 1
        cores_per_fit = min(chains, available_cores)
        sampler = "blackjax" if analysis == "rsf" else "nutpie"
    else:
        sampler = "nutpie"
        if profile == "coolmuc4" and available_cores == 112 and chains >= 4:
            measured_outer_cap = 56 if analysis == "rsf" else 28
            outer_workers = min(n_models, measured_outer_cap)
            cores_per_fit = min(
                chains,
                max(1, available_cores // outer_workers),
            )
        else:
            outer_workers = min(n_models, available_cores)
            cores_per_fit = min(
                chains,
                max(1, available_cores // outer_workers),
            )

    active_cores = outer_workers * cores_per_fit
    return BayesianCPUPlan(
        analysis=analysis,
        objective=resolved_objective,
        sampler=sampler,
        n_models=int(n_models),
        available_cores=int(available_cores),
        chains=int(chains),
        outer_workers=int(outer_workers),
        cores_per_fit=int(cores_per_fit),
        active_cores=int(active_cores),
        unused_cores=int(max(0, available_cores - active_cores)),
        profile=profile,
    )


__all__ = ["BayesianCPUPlan", "plan_bayesian_cpu_execution"]
