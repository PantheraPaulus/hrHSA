"""Validate Bayesian SSF/iSSF samplers in fresh Python processes.

Each sampler/workload/repeat gets a new interpreter. Completed sampling includes
explicit host materialization, and compact posterior summaries are compared with a
reference sampler on the same synthetic choice experiment and sampling seed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from validate_bayesian_samplers import _csv_strings, _posterior_agreement


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "benchmark_bayesian_ssf_completion.py"


def _positive_ints(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out or any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return "<not created>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fresh-process completion and posterior validation for Bayesian SSF/iSSF."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis", choices=("ssf", "issf"), required=True)
    parser.add_argument("--strata", type=_positive_ints, default=[10_000, 50_000, 100_000])
    parser.add_argument("--choices", type=int, default=11)
    parser.add_argument("--predictors", type=int, default=6)
    parser.add_argument("--individuals", type=int, default=20)
    parser.add_argument("--samplers", default="pymc,blackjax")
    parser.add_argument("--reference-sampler", default="pymc")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--blas-cores", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=10_000)
    parser.add_argument("--memory-sample-interval", type=float, default=0.25)
    args = parser.parse_args()

    samplers = _csv_strings(args.samplers)
    if args.reference_sampler not in samplers:
        parser.error("--reference-sampler must be included in --samplers")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.analysis == "issf" and args.predictors < 3:
        parser.error("iSSF requires at least three predictors")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    summaries: dict[tuple[int, int, str], Path] = {}

    for n_strata in args.strata:
        for repeat in range(1, args.repeats + 1):
            sampling_seed = args.sampling_seed + repeat
            for sampler in samplers:
                child_dir = output_dir / f"strata-{n_strata:09d}" / f"repeat-{repeat:02d}" / sampler
                child_dir.mkdir(parents=True, exist_ok=True)
                result_path = child_dir / "completion.json"
                summary_path = child_dir / "posterior_summary.csv"
                stdout_path = child_dir / "stdout.log"
                stderr_path = child_dir / "stderr.log"

                command = [
                    sys.executable,
                    str(BENCHMARK),
                    "--analysis",
                    args.analysis,
                    "--sampler",
                    sampler,
                    "--output",
                    str(result_path),
                    "--posterior-summary",
                    str(summary_path),
                    "--strata",
                    str(n_strata),
                    "--choices",
                    str(args.choices),
                    "--predictors",
                    str(args.predictors),
                    "--individuals",
                    str(args.individuals),
                    "--draws",
                    str(args.draws),
                    "--tune",
                    str(args.tune),
                    "--chains",
                    str(args.chains),
                    "--cores",
                    str(args.cores),
                    "--target-accept",
                    str(args.target_accept),
                    "--data-seed",
                    str(args.data_seed + n_strata),
                    "--sampling-seed",
                    str(sampling_seed),
                    "--memory-sample-interval",
                    str(args.memory_sample_interval),
                ]
                if args.blas_cores is not None:
                    command.extend(["--blas-cores", str(args.blas_cores)])

                env = os.environ.copy()
                if "JAX_COMPILATION_CACHE_DIR" in env:
                    env["JAX_COMPILATION_CACHE_DIR"] = str(child_dir / "jax-compilation-cache")

                started = perf_counter()
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                    "w", encoding="utf-8"
                ) as stderr:
                    completed = subprocess.run(
                        command,
                        cwd=HERE.parents[1],
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        check=False,
                    )
                process_seconds = perf_counter() - started
                result = _read_json(result_path)
                success = completed.returncode == 0 and result.get("status") == "success"

                row = {
                    "analysis": args.analysis,
                    "n_strata": n_strata,
                    "sampler": sampler,
                    "repeat": repeat,
                    "status": "success" if success else "failed",
                    "process_seconds": process_seconds,
                    "returncode": completed.returncode,
                    "prepare_seconds": result.get("prepare_seconds", np.nan),
                    "model_seconds": result.get("model_seconds", np.nan),
                    "dispatch_seconds": result.get("dispatch_seconds", np.nan),
                    "materialize_seconds": result.get("materialize_seconds", np.nan),
                    "completed_sampling_seconds": result.get("completed_sampling_seconds", np.nan),
                    "diagnostics_seconds": result.get("diagnostics_seconds", np.nan),
                    "median_ess_bulk": result.get("median_ess_bulk", np.nan),
                    "min_ess_bulk": result.get("min_ess_bulk", np.nan),
                    "median_ess_bulk_per_completed_second": result.get(
                        "median_ess_bulk_per_completed_second", np.nan
                    ),
                    "min_ess_bulk_per_completed_second": result.get(
                        "min_ess_bulk_per_completed_second", np.nan
                    ),
                    "median_ess_tail": result.get("median_ess_tail", np.nan),
                    "min_ess_tail": result.get("min_ess_tail", np.nan),
                    "median_ess_tail_per_completed_second": result.get(
                        "median_ess_tail_per_completed_second", np.nan
                    ),
                    "max_rhat": result.get("max_rhat", np.nan),
                    "n_divergences": result.get("n_divergences", np.nan),
                    "mean_n_steps": result.get("mean_n_steps", np.nan),
                    "max_tree_depth": result.get("max_tree_depth", np.nan),
                    "operation_peak_process_tree_rss_mb": result.get(
                        "operation_peak_process_tree_rss_mb", np.nan
                    ),
                    "posterior_mib": result.get("posterior_mib", np.nan),
                    "offset": result.get("offset"),
                    "offset_sd": result.get("offset_sd", np.nan),
                    "sampling_seed": sampling_seed,
                    "posterior_summary": result.get("posterior_summary"),
                }
                run_rows.append(row)

                if success and summary_path.exists():
                    summaries[(n_strata, repeat, sampler)] = summary_path
                else:
                    print(
                        f"[FAILED] {args.analysis} strata={n_strata} sampler={sampler} "
                        f"repeat={repeat} returncode={completed.returncode}\n"
                        f"stderr tail:\n{_tail(stderr_path)}"
                    )

    runs = pd.DataFrame(run_rows)
    runs.to_csv(output_dir / "sampler_completion_runs.csv", index=False)

    comparisons: list[dict[str, Any]] = []
    for n_strata in args.strata:
        for repeat in range(1, args.repeats + 1):
            ref_path = summaries.get((n_strata, repeat, args.reference_sampler))
            ref_rows = runs.loc[
                (runs["n_strata"] == n_strata)
                & (runs["repeat"] == repeat)
                & (runs["sampler"] == args.reference_sampler)
                & (runs["status"] == "success")
            ]
            if ref_path is None or ref_rows.empty:
                continue
            ref = ref_rows.iloc[0]
            for sampler in samplers:
                if sampler == args.reference_sampler:
                    continue
                candidate_path = summaries.get((n_strata, repeat, sampler))
                candidate_rows = runs.loc[
                    (runs["n_strata"] == n_strata)
                    & (runs["repeat"] == repeat)
                    & (runs["sampler"] == sampler)
                    & (runs["status"] == "success")
                ]
                if candidate_path is None or candidate_rows.empty:
                    continue
                candidate = candidate_rows.iloc[0]
                ref_seconds = float(ref["completed_sampling_seconds"])
                cand_seconds = float(candidate["completed_sampling_seconds"])
                ref_ess_s = float(ref["median_ess_bulk_per_completed_second"])
                cand_ess_s = float(candidate["median_ess_bulk_per_completed_second"])
                comparisons.append(
                    {
                        "analysis": args.analysis,
                        "n_strata": n_strata,
                        "repeat": repeat,
                        "reference_sampler": args.reference_sampler,
                        "sampler": sampler,
                        "completed_speedup_vs_reference": ref_seconds / cand_seconds,
                        "completed_ess_s_ratio_vs_reference": cand_ess_s / ref_ess_s,
                        "reference_completed_seconds": ref_seconds,
                        "candidate_completed_seconds": cand_seconds,
                        **_posterior_agreement(ref_path, candidate_path),
                    }
                )

    agreement = pd.DataFrame(comparisons)
    agreement.to_csv(output_dir / "posterior_completion_agreement.csv", index=False)

    print("\nCompleted fresh-process Bayesian SSF/iSSF runs")
    print(runs.to_string(index=False))
    if not agreement.empty:
        print("\nPosterior agreement and completed-time comparison")
        print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()
