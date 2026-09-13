"""Print hrHSA's explicit CPU execution recommendation for Bayesian workloads."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from hsa.compute import plan_bayesian_cpu_execution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan outer-model and inner-chain CPU concurrency for Bayesian hrHSA fits."
    )
    parser.add_argument("--analysis", choices=("rsf", "ssf", "issf"), required=True)
    parser.add_argument("--models", type=int, required=True)
    parser.add_argument("--available-cores", type=int, required=True)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--objective",
        choices=("auto", "latency", "throughput"),
        default="auto",
    )
    parser.add_argument(
        "--profile",
        choices=("generic", "coolmuc4"),
        default="generic",
        help="Optional measured site profile; default preserves portable outer-first behavior.",
    )
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()

    plan = plan_bayesian_cpu_execution(
        analysis=args.analysis,
        n_models=args.models,
        available_cores=args.available_cores,
        chains=args.chains,
        objective=args.objective,
        profile=args.profile,
    )

    if args.format == "json":
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        return

    values = {
        "HRHSA_BAYES_ANALYSIS": plan.analysis,
        "HRHSA_BAYES_OBJECTIVE": plan.objective,
        "HRHSA_BAYES_PROFILE": plan.profile,
        "HRHSA_BAYES_SAMPLER": plan.sampler,
        "HRHSA_BAYES_MODELS": plan.n_models,
        "HRHSA_BAYES_AVAILABLE_CORES": plan.available_cores,
        "HRHSA_BAYES_CHAINS": plan.chains,
        "HRHSA_BAYES_OUTER_WORKERS": plan.outer_workers,
        "HRHSA_BAYES_CORES_PER_FIT": plan.cores_per_fit,
        "HRHSA_BAYES_ACTIVE_CORES": plan.active_cores,
        "HRHSA_BAYES_UNUSED_CORES": plan.unused_cores,
    }
    for name, value in values.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
